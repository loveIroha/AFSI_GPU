#include <ATen/ATen.h>
#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAGuard.h>
#include <c10/cuda/CUDAException.h>
#include <cuda_runtime.h>
#include <climits>
#include <type_traits>
using at::Tensor;

// No characteristic, quadrature, mass, or transfer equation is changed.
// The shared layout stores the same integer/half-integer one-dimensional tables.
template<class T>
__global__ void prepare_kernel(const T* x,const int64_t* cells,const T* N,const T* origin,
    const T* spacing,const T* limits,int64_t* base,T* phi,int* invalid,
    int64_t ne,int64_t nq,int64_t total,int64_t offset,int64_t nodes) {
    int64_t lane=int64_t(blockIdx.x)*blockDim.x+threadIdx.x;
    if(lane>=ne*nq*3)return;
    int axis=lane%3;
    int64_t p=lane/3,e=p/nq,q=p%nq;
    T position=0;
    #pragma unroll
    for(int a=0;a<10;++a) {
        int64_t node=cells[e*10+a];
        CUDA_KERNEL_ASSERT(node>=0 && node<nodes);
        if(node<0 || node>=nodes)return;
        position+=N[q*10+a]*x[node*3+axis];
    }
    T physical=(position-origin[axis])/spacing[axis];
    bool valid=isfinite(physical) && physical>=T(2) && physical<limits[axis];
    if(!valid)atomicOr(invalid,1);
    #pragma unroll
    for(int lattice=0;lattice<2;++lattice) {
        T start=lattice?origin[axis]+T(.5)*spacing[axis]:origin[axis];
        T scaled=valid?(position-start)/spacing[axis]:T(0);
        int64_t lower=int64_t(floor(scaled-T(1)));
        T f=scaled-T(lower)-T(1);
        T radical=sqrt(T(1)+T(4)*f-T(4)*(f*f));
        int64_t index=(lattice*total+offset+p)*3+axis;
        base[index]=lower;
        phi[index*4]=(T(3)-T(2)*f-radical)/T(8);
        phi[index*4+1]=(T(3)-T(2)*f+radical)/T(8);
        phi[index*4+2]=(T(1)+T(2)*f+radical)/T(8);
        phi[index*4+3]=(T(1)+T(2)*f-radical)/T(8);
    }
}

constexpr int THREADS=256,WARPS=THREADS/32,CACHE=2048;
template<class T>struct Patch {
    int low[6],high[6],start[3][3],extent[3][3],offset[4];
    T values[CACHE],sums[WARPS*30];
};

template<class T>
__device__ void plan(Patch<T>& s,const int64_t* base,int64_t first,int64_t nq,int64_t total) {
    int tid=threadIdx.x,lane=tid%32;
    if(tid<6){s.low[tid]=INT_MAX;s.high[tid]=INT_MIN;}
    __syncthreads();
    #pragma unroll
    for(int a=0;a<6;++a) {
        int low=INT_MAX,high=INT_MIN;
        for(int64_t q=tid;q<nq;q+=THREADS) {
            int v=int(base[((a/3)*total+first+q)*3+a%3]);
            low=min(low,v);high=max(high,v);
        }
        for(int d=16;d;d/=2){low=min(low,__shfl_down_sync(0xffffffff,low,d));high=max(high,__shfl_down_sync(0xffffffff,high,d));}
        if(lane==0){atomicMin(s.low+a,low);atomicMax(s.high+a,high);}
    }
    __syncthreads();
    if(tid==0) {
        s.offset[0]=0;
        for(int c=0;c<3;++c) {
            int size=1;
            for(int a=0;a<3;++a) {
                int j=3*int(a!=c)+a;
                s.start[c][a]=s.low[j];s.extent[c][a]=s.high[j]-s.low[j]+4;
                size*=s.extent[c][a];
            }
            s.offset[c+1]=s.offset[c]+size;
        }
    }
    __syncthreads();
}
template<class T>
__device__ int patch_index(const Patch<T>& s,int c,int x,int y,int z) {
    return s.offset[c]+((x-s.start[c][0])*s.extent[c][1]+y-s.start[c][1])*s.extent[c][2]+z-s.start[c][2];
}
__device__ int64_t grid_index(int c,int x,int y,int z,int64_t ny,int64_t nz) {
    return (int64_t(x)*(ny+int(c==1))+y)*(nz+int(c==2))+z;
}
template<class T>
__device__ T coefficient(const T* phi,int64_t p,int64_t total,int c,int i,int j,int k) {
    auto x=((int(c!=0)*total+p)*3)*4+i;
    auto y=((int(c!=1)*total+p)*3+1)*4+j;
    auto z=((int(c!=2)*total+p)*3+2)*4+k;
    return (phi[x]*phi[y])*phi[z];
}

template<class T,bool Spread>
__global__ void transfer_kernel(const T* __restrict__ value,T* __restrict__ u0,T* __restrict__ u1,T* __restrict__ u2,const int64_t* __restrict__ cells,
    const T* __restrict__ N,const T* __restrict__ W,const int64_t* __restrict__ base,const T* __restrict__ phi,T* __restrict__ rhs,
    int64_t nq,int64_t total,int64_t offset,int64_t nx,int64_t ny,int64_t nz,int64_t nodes,T volume) {
    __shared__ Patch<T> s;
    const int tid=threadIdx.x,lane=tid%32,warp=tid/32;
    const int64_t e=blockIdx.x,first=offset+e*nq;
    plan(s,base,first,nq,total);
    // Uniform block predicate: malformed public-operator tables must never
    // turn into unchecked grid or node memory reads.
    bool valid=true;
    for(int c=0;c<3;++c)for(int a=0;a<3;++a) {
        int64_t extent=(a==0?nx:a==1?ny:nz)+int(a==c);
        valid=valid && s.start[c][a]>=0 && int64_t(s.start[c][a])+s.extent[c][a]<=extent;
    }
    for(int a=0;a<10;++a)valid=valid && cells[e*10+a]>=0 && cells[e*10+a]<nodes;
    CUDA_KERNEL_ASSERT(valid);
    if(!valid)return;
    bool cached=s.offset[3]<=CACHE;
    if(cached) {
        for(int i=tid;i<s.offset[3];i+=THREADS) {
            if constexpr(Spread)s.values[i]=T(0);
            else {
                int c=i>=s.offset[2]?2:i>=s.offset[1]?1:0;
                int p=i-s.offset[c],z=p%s.extent[c][2]+s.start[c][2];
                p/=s.extent[c][2];int y=p%s.extent[c][1]+s.start[c][1],x=p/s.extent[c][1]+s.start[c][0];
                const T* field=c==0?u0:c==1?u1:u2;
                s.values[i]=field[grid_index(c,x,y,z,ny,nz)];
            }
        }
    }
    __syncthreads();
    T sums[3];
    if constexpr(!Spread) {
        #pragma unroll
        for(int a=0;a<3;++a)sums[a]=T(0);
    }
    for(int64_t q=warp;q<nq;q+=WARPS) {
        int64_t p=first+q;
        #pragma unroll
        for(int c=0;c<3;++c) {
            T force=0,got=0;
            if constexpr(Spread) {
                if(lane<10)force=N[q*10+lane]*value[cells[e*10+lane]*3+c];
                for(int d=16;d;d/=2)force+=__shfl_down_sync(0xffffffff,force,d);
                force=__shfl_sync(0xffffffff,force,0)*W[e*nq+q];
            }
            int bx=int(base[(int(c!=0)*total+p)*3]);
            int by=int(base[(int(c!=1)*total+p)*3+1]);
            int bz=int(base[(int(c!=2)*total+p)*3+2]);
            #pragma unroll
            for(int k=lane;k<64;k+=32) {
                int i=k/16,j=(k/4)%4,l=k%4,x=bx+i,y=by+j,z=bz+l;
                T weight=coefficient(phi,p,total,c,i,j,l);
                T* field=c==0?u0:c==1?u1:u2;
                if constexpr(Spread) {
                    T contribution=(weight*force)/volume;
                    if(cached)atomicAdd(s.values+patch_index(s,c,x,y,z),contribution);
                    else atomicAdd(field+grid_index(c,x,y,z,ny,nz),contribution);
                } else {
                    T v=cached?s.values[patch_index(s,c,x,y,z)]:field[grid_index(c,x,y,z,ny,nz)];
                    got+=v*weight;
                }
            }
            if constexpr(!Spread) {
                for(int d=16;d;d/=2)got+=__shfl_down_sync(0xffffffff,got,d);
                got=__shfl_sync(0xffffffff,got,0);
                if(lane<10)sums[c]+=(N[q*10+lane]*W[e*nq+q])*got;
            }
        }
    }
    if constexpr(Spread) {
        __syncthreads();
        if(cached)for(int i=tid;i<s.offset[3];i+=THREADS) {
            int c=i>=s.offset[2]?2:i>=s.offset[1]?1:0;
            int p=i-s.offset[c],z=p%s.extent[c][2]+s.start[c][2];
            p/=s.extent[c][2];int y=p%s.extent[c][1]+s.start[c][1],x=p/s.extent[c][1]+s.start[c][0];
            T* field=c==0?u0:c==1?u1:u2;
            atomicAdd(field+grid_index(c,x,y,z,ny,nz),s.values[i]);
        }
    } else {
        if(lane<10)for(int c=0;c<3;++c)s.sums[warp*30+lane*3+c]=sums[c];
        __syncthreads();
        if(tid<30) {
            T sum=0;
            #pragma unroll
            for(int w=0;w<WARPS;++w)sum+=s.sums[w*30+tid];
            atomicAdd(rhs+cells[e*10+tid/3]*3+tid%3,sum);
        }
    }
}

void p2_prepare_cuda(const Tensor& x,const Tensor& cells,const Tensor& N,const Tensor& origin,
    const Tensor& spacing,const Tensor& limits,const Tensor& base,const Tensor& phi,
    const Tensor& invalid,int64_t offset) {
    const c10::cuda::CUDAGuard guard(x.device());
    int64_t count=cells.size(0)*N.size(0)*3;
    if(!count)return;
    AT_DISPATCH_FLOATING_TYPES(x.scalar_type(),"afsi_p2_prepare",[&]{
        prepare_kernel<scalar_t><<<(count+255)/256,256,0,at::cuda::getCurrentCUDAStream()>>>(
            x.data_ptr<scalar_t>(),cells.data_ptr<int64_t>(),N.data_ptr<scalar_t>(),origin.data_ptr<scalar_t>(),
            spacing.data_ptr<scalar_t>(),limits.data_ptr<scalar_t>(),base.data_ptr<int64_t>(),phi.data_ptr<scalar_t>(),
            invalid.data_ptr<int>(),cells.size(0),N.size(0),base.size(1),offset,x.size(0));
    });
    C10_CUDA_KERNEL_LAUNCH_CHECK();
}
void p2_transfer_cuda(const Tensor& value,const Tensor& u0,const Tensor& u1,const Tensor& u2,
    const Tensor& cells,const Tensor& N,const Tensor& W,const Tensor& base,const Tensor& phi,const Tensor& rhs,
    int64_t offset,int64_t nx,int64_t ny,int64_t nz,double volume,bool spread) {
    if(!cells.size(0))return;
    const c10::cuda::CUDAGuard guard(value.device());
    AT_DISPATCH_FLOATING_TYPES(value.scalar_type(),"afsi_p2_transfer",[&]{
        const auto launch=[&](auto mode) {
            transfer_kernel<scalar_t,decltype(mode)::value><<<cells.size(0),THREADS,0,at::cuda::getCurrentCUDAStream()>>>(
                value.data_ptr<scalar_t>(),u0.data_ptr<scalar_t>(),u1.data_ptr<scalar_t>(),u2.data_ptr<scalar_t>(),
                cells.data_ptr<int64_t>(),N.data_ptr<scalar_t>(),W.data_ptr<scalar_t>(),base.data_ptr<int64_t>(),
                phi.data_ptr<scalar_t>(),rhs.data_ptr<scalar_t>(),N.size(0),base.size(1),offset,nx,ny,nz,value.size(0),scalar_t(volume));
        };
        if(spread)launch(std::true_type{});else launch(std::false_type{});
    });
    C10_CUDA_KERNEL_LAUNCH_CHECK();
}
