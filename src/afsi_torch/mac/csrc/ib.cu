#include <ATen/ATen.h>
#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAGuard.h>
#include <c10/cuda/CUDAException.h>
#include <cuda_runtime.h>
#include <algorithm>
#include <cstdint>
#include <type_traits>

using at::Tensor;

__device__ __forceinline__ int64_t broadcast(int64_t v) {
    return static_cast<int64_t>(__shfl_sync(0xffffffff,static_cast<long long>(v),0));
}

__device__ __forceinline__ uint64_t bucket(int64_t key, int64_t capacity) {
    uint64_t h=static_cast<uint64_t>(key);
    h=(h^(h>>30))*0xbf58476d1ce4e5b9ULL;
    h=(h^(h>>27))*0x94d049bb133111ebULL;
    return (h^(h>>31)) & (capacity-1);
}

template<typename T, bool Cached>
__device__ __forceinline__ void add(int64_t key,T value,int64_t* keys,T* values,
    int32_t* flag,int64_t capacity,int64_t* mkeys,T* mvalues,int64_t* mcount,int64_t mcap) {
    if (value==T(0)) return;
    if (key<0) { atomicExch(flag,1); return; }
    auto slot=bucket(key,capacity);
    for (int attempt=0;attempt<128;++attempt,slot=(slot+1)&(capacity-1)) {
        int64_t previous;
        if constexpr (Cached) {
            // Entire key table is immutable until this launch completes.
            previous=keys[slot];
        } else {
            previous=static_cast<int64_t>(atomicCAS(
                reinterpret_cast<unsigned long long*>(keys+slot),
                static_cast<unsigned long long>(-1LL),static_cast<unsigned long long>(key)));
        }
        if (previous==key || (!Cached && previous==-1)) {
            atomicAdd(values+slot,value);
            return;
        }
        if constexpr (Cached) { if (previous==-1) break; }
    }
    if constexpr (Cached) {
        auto i=atomicAdd(reinterpret_cast<unsigned long long*>(mcount),1ULL);
        if (i<static_cast<uint64_t>(mcap)) { mkeys[i]=key; mvalues[i]=value; }
        // Count every miss, including overflow: caller discards/rebuilds.
    } else {
        atomicExch(flag,1);
    }
}

// A warp cooperates on one cell/lattice pair. Quadrature data feed all four
// P1 nodal accumulators before being discarded; no raw Q*64*4 entry stream.
template<typename T,bool Cached,bool Shared>
__global__ void contract_kernel(const int64_t* base,const T* phi,const T* shape,
    const T* weights,const int64_t* cells,const int64_t* low,const int64_t* width,
    const int64_t* prefix,int64_t* keys,T* values,int32_t* flag,int64_t* mkeys,
    T* mvalues,int64_t* mcount,int64_t mcap,int64_t capacity,int64_t offset,
    int64_t count,int64_t p,int64_t q,int64_t e,int c,int64_t ny,int64_t nz,int64_t nf) {
    const int lane=threadIdx.x&31;
    const int64_t warp=(int64_t(blockIdx.x)*blockDim.x+threadIdx.x)/32;
    const int64_t stride=int64_t(gridDim.x)*blockDim.x/32;
    for (int64_t site=warp;site<count;site+=stride) {
        int64_t cell=0,gx=0,gy=0,gz=0;
        if (lane==0) {
            int64_t left=0,right=e;
            while (left<right) {
                auto middle=(left+right)/2;
                if (prefix[middle+1]<=site) left=middle+1; else right=middle;
            }
            cell=left;
            // Prefix values are produced by component_plans; reject malformed
            // coverage on device instead of accessing outside the cell arrays.
            if (cell<e) {
                auto relative=site-prefix[cell],sy=width[3*cell+1],sz=width[3*cell+2];
                if (sy<=0 || sz<=0) cell=e;
                else {
                    gx=relative/(sy*sz)+low[3*cell];
                    gy=(relative/sz)%sy+low[3*cell+1];
                    gz=relative%sz+low[3*cell+2];
                }
            }
        }
        cell=broadcast(cell);
        gx=broadcast(gx); gy=broadcast(gy); gz=broadcast(gz);
        if (cell>=e || gx<0 || gy<0 || gz<0 || gx>=nf/(ny*nz) || gy>=ny || gz>=nz) {
            if (lane==0) atomicExch(flag,1);
            continue;
        }
        T sums[4]={T(0),T(0),T(0),T(0)};
        const int lx=Shared?(c==0?0:1):c,ly=Shared?(c==1?0:1):c,lz=Shared?(c==2?0:1):c;
        for (int64_t i=lane;i<q;i+=32) {
            const auto point=offset+cell*q+i;
            const auto ix=gx-base[(lx*p+point)*3],iy=gy-base[(ly*p+point)*3+1],iz=gz-base[(lz*p+point)*3+2];
            if (ix<0 || ix>=4 || iy<0 || iy>=4 || iz<0 || iz>=4) continue;
            T k=((phi[((lx*p+point)*3)*4+ix]*phi[((ly*p+point)*3+1)*4+iy])
                 *phi[((lz*p+point)*3+2)*4+iz])*weights[cell*q+i];
            # --fmad=false preserves separate products/additions in FP64/FP32.
            # Different parallel reduction orders still permit roundoff.
            #pragma unroll
            for (int a=0;a<4;++a) sums[a]+=k*shape[i*4+a];
        }
        #pragma unroll
        for (int a=0;a<4;++a) {
            for (int distance=16;distance>0;distance/=2)
                sums[a]+=__shfl_down_sync(0xffffffff,sums[a],distance);
            sums[a]=__shfl_sync(0xffffffff,sums[a],0);
        }
        if (lane<4) {
            auto key=cells[cell*4+lane]*nf+(gx*ny+gy)*nz+gz;
            add<T,Cached>(key,sums[lane],keys,values,flag,capacity,mkeys,mvalues,mcount,mcap);
        }
    }
}

template<typename T>
__global__ void accumulate_kernel(const int64_t* input_keys,const T* input_values,
    int64_t count,int64_t* keys,T* values,int32_t* flag,int64_t capacity) {
    for (int64_t i=int64_t(blockIdx.x)*blockDim.x+threadIdx.x;i<count;i+=int64_t(gridDim.x)*blockDim.x)
        add<T,false>(input_keys[i],input_values[i],keys,values,flag,capacity,nullptr,nullptr,nullptr,0);
}

void ib_contract_cuda(const Tensor& base,const Tensor& phi,const Tensor& shape,
    const Tensor& weights,const Tensor& cells,const Tensor& low,const Tensor& width,
    const Tensor& prefix,const Tensor& keys,const Tensor& values,const Tensor& flag,
    const Tensor& mkeys,const Tensor& mvalues,const Tensor& mcount,int64_t offset,
    int64_t count,int64_t c,int64_t ny,int64_t nz,int64_t nf,bool shared,bool cached) {
    if (!count) return;
    const c10::cuda::CUDAGuard guard(values.device());
    const auto stream=at::cuda::getCurrentCUDAStream(values.get_device());
    const int blocks=static_cast<int>(std::min<int64_t>((count+3)/4,65535));
    AT_DISPATCH_FLOATING_TYPES(values.scalar_type(),"afsi_ib_contract",[&] {
        const auto launch=[&](auto cache_tag,auto shared_tag) {
            constexpr bool ca=decltype(cache_tag)::value,sh=decltype(shared_tag)::value;
            contract_kernel<scalar_t,ca,sh><<<blocks,128,0,stream>>>(
                base.data_ptr<int64_t>(),phi.data_ptr<scalar_t>(),shape.data_ptr<scalar_t>(),
                weights.data_ptr<scalar_t>(),cells.data_ptr<int64_t>(),low.data_ptr<int64_t>(),
                width.data_ptr<int64_t>(),prefix.data_ptr<int64_t>(),keys.data_ptr<int64_t>(),
                values.data_ptr<scalar_t>(),flag.data_ptr<int32_t>(),
                cached?mkeys.data_ptr<int64_t>():nullptr,cached?mvalues.data_ptr<scalar_t>():nullptr,
                cached?mcount.data_ptr<int64_t>():nullptr,cached?mkeys.numel():0,keys.numel(),
                offset,count,base.size(1),shape.size(0),cells.size(0),static_cast<int>(c),ny,nz,nf);
        };
        if (cached) {
            if (shared) launch(std::true_type{},std::true_type{});
            else launch(std::true_type{},std::false_type{});
        } else {
            if (shared) launch(std::false_type{},std::true_type{});
            else launch(std::false_type{},std::false_type{});
        }
    });
    C10_CUDA_KERNEL_LAUNCH_CHECK();
}

void ib_accumulate_cuda(const Tensor& input_keys,const Tensor& input_values,
    const Tensor& keys,const Tensor& values,const Tensor& flag) {
    if (!input_keys.numel()) return;
    const c10::cuda::CUDAGuard guard(values.device());
    const auto stream=at::cuda::getCurrentCUDAStream(values.get_device());
    const int blocks=static_cast<int>(std::min<int64_t>((input_keys.numel()+127)/128,65535));
    AT_DISPATCH_FLOATING_TYPES(values.scalar_type(),"afsi_ib_accumulate",[&] {
        accumulate_kernel<scalar_t><<<blocks,128,0,stream>>>(input_keys.data_ptr<int64_t>(),
            input_values.data_ptr<scalar_t>(),input_keys.numel(),keys.data_ptr<int64_t>(),
            values.data_ptr<scalar_t>(),flag.data_ptr<int32_t>(),keys.numel());
    });
    C10_CUDA_KERNEL_LAUNCH_CHECK();
}
