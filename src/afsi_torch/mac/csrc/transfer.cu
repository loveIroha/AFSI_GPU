#include <ATen/ATen.h>
#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAGuard.h>
#include <c10/cuda/CUDAException.h>
#include <cuda_runtime.h>
using at::Tensor;

// One warp per point/component. Signed wall-reflection weights and duplicate
// indices are used identically in the transpose. No P*K*C temporary tensor.
template<class T,bool Spread>
__global__ void indexed_kernel(const T* f,const int64_t* ids,const T* w,T* out,
        int64_t points,int64_t neighbors,int64_t components,int64_t nodes,T volume) {
    const int lane=threadIdx.x%32;
    const int64_t task=(int64_t(blockIdx.x)*blockDim.x+threadIdx.x)/32;
    if(task>=points*components) return;
    const int64_t p=task/components,c=task%components;
    T sum=0;
    for(int64_t k=lane;k<neighbors;k+=32) {
        const int64_t j=ids[p*neighbors+k];
        if(j<0 || j>=nodes) { CUDA_KERNEL_ASSERT(false); continue; }
        const T weight=w[p*neighbors+k];
        if constexpr(Spread) atomicAdd(out+j*components+c,(f[p*components+c]*weight)/volume);
        else sum+=f[j*components+c]*weight;
    }
    if constexpr(!Spread) {
        for(int d=16;d;d/=2) sum+=__shfl_down_sync(0xffffffff,sum,d);
        if(lane==0) out[p*components+c]=sum;
    }
}

// Compact three-dimensional MAC stencils stay separable; the 64 neighbors
// are reconstructed in registers rather than expanded in global memory.
template<class T,bool Spread>
__global__ void compact_kernel(const T* f,const int64_t* base,const T* phi,T* out,
        int64_t points,int64_t nx,int64_t ny,int64_t nz,T volume) {
    const int lane=threadIdx.x%32;
    const int64_t p=(int64_t(blockIdx.x)*blockDim.x+threadIdx.x)/32;
    if(p>=points) return;
    T sum=0;
    for(int k=lane;k<64;k+=32) {
        const int i=k/16,j=(k/4)%4,l=k%4;
        const int64_t x=base[3*p]+i,y=base[3*p+1]+j,z=base[3*p+2]+l;
        if(x<0||x>=nx||y<0||y>=ny||z<0||z>=nz) { CUDA_KERNEL_ASSERT(false); continue; }
        const int64_t id=(x*ny+y)*nz+z;
        const T w=(phi[p*12+i]*phi[p*12+4+j])*phi[p*12+8+l];
        if constexpr(Spread) atomicAdd(out+id,(w*f[p])/volume);
        else sum+=f[id]*w;
    }
    if constexpr(!Spread) {
        for(int d=16;d;d/=2) sum+=__shfl_down_sync(0xffffffff,sum,d);
        if(lane==0) out[p]=sum;
    }
}

Tensor ib_indexed_cuda(const Tensor& f,const Tensor& ids,const Tensor& w,int64_t size,double volume,bool spread) {
    const c10::cuda::CUDAGuard guard(f.device());
    const auto p=ids.size(0),k=ids.size(1),c=f.size(1),n=spread?size:f.size(0);
    auto out=spread?at::zeros({size,c},f.options()):at::empty({p,c},f.options());
    if(!p) return out;
    const auto stream=at::cuda::getCurrentCUDAStream();
    const int blocks=int((p*c+7)/8);
    AT_DISPATCH_FLOATING_TYPES(f.scalar_type(),"ib_indexed",[&] {
        if(spread) indexed_kernel<scalar_t,true><<<blocks,256,0,stream>>>(f.data_ptr<scalar_t>(),ids.data_ptr<int64_t>(),w.data_ptr<scalar_t>(),out.data_ptr<scalar_t>(),p,k,c,n,scalar_t(volume));
        else indexed_kernel<scalar_t,false><<<blocks,256,0,stream>>>(f.data_ptr<scalar_t>(),ids.data_ptr<int64_t>(),w.data_ptr<scalar_t>(),out.data_ptr<scalar_t>(),p,k,c,n,scalar_t(volume));
    });
    C10_CUDA_KERNEL_LAUNCH_CHECK();
    return out;
}
Tensor ib_compact_cuda(const Tensor& f,const Tensor& base,const Tensor& phi,int64_t nx,int64_t ny,int64_t nz,double volume,bool spread) {
    const c10::cuda::CUDAGuard guard(f.device());
    const auto p=base.size(0);
    auto out=spread?at::zeros({nx*ny*nz},f.options()):at::empty({p},f.options());
    if(!p) return out;
    const auto stream=at::cuda::getCurrentCUDAStream();
    const int blocks=int((p+7)/8);
    AT_DISPATCH_FLOATING_TYPES(f.scalar_type(),"ib_compact",[&] {
        if(spread) compact_kernel<scalar_t,true><<<blocks,256,0,stream>>>(f.data_ptr<scalar_t>(),base.data_ptr<int64_t>(),phi.data_ptr<scalar_t>(),out.data_ptr<scalar_t>(),p,nx,ny,nz,scalar_t(volume));
        else compact_kernel<scalar_t,false><<<blocks,256,0,stream>>>(f.data_ptr<scalar_t>(),base.data_ptr<int64_t>(),phi.data_ptr<scalar_t>(),out.data_ptr<scalar_t>(),p,nx,ny,nz,scalar_t(volume));
    });
    C10_CUDA_KERNEL_LAUNCH_CHECK();
    return out;
}
