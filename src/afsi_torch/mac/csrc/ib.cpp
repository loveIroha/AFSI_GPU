#include <ATen/ATen.h>
#include <torch/library.h>
#include <limits>

using at::Tensor;

void ib_contract_cuda(const Tensor&, const Tensor&, const Tensor&, const Tensor&,
                     const Tensor&, const Tensor&, const Tensor&, const Tensor&,
                     const Tensor&, const Tensor&, const Tensor&, const Tensor&,
                     const Tensor&, const Tensor&, int64_t, int64_t, int64_t,
                     int64_t, int64_t, int64_t, bool, bool);
void ib_accumulate_cuda(const Tensor&, const Tensor&, const Tensor&, const Tensor&, const Tensor&);

static void check(const Tensor& t, const Tensor& like, at::ScalarType dtype) {
    TORCH_CHECK(t.is_cuda() && t.device()==like.device(), "IB tensors must share a CUDA device");
    TORCH_CHECK(t.is_contiguous(), "IB tensors must be contiguous");
    TORCH_CHECK(t.scalar_type()==dtype, "IB tensor dtype mismatch");
    TORCH_CHECK(!t.requires_grad(), "IB construction is a frozen-geometry operator; gradients are not registered");
}

static void check_table(const Tensor& keys, const Tensor& values, const Tensor& flag) {
    TORCH_CHECK(values.scalar_type()==at::kFloat || values.scalar_type()==at::kDouble,
                "IB supports float32/float64");
    check(values, values, values.scalar_type());
    check(keys, values, at::kLong);
    check(flag, values, at::kInt);
    TORCH_CHECK(keys.dim()==1 && values.dim()==1 && keys.numel()==values.numel(), "IB hash shape mismatch");
    const auto n=keys.numel();
    TORCH_CHECK(n>0 && (n & (n-1))==0, "IB requires power-of-two hash capacity");
    TORCH_CHECK(flag.numel()==1, "IB flag must contain one int32");
}

static void check_inputs(const Tensor& base, const Tensor& phi, const Tensor& shape,
                         const Tensor& weights, const Tensor& cells, const Tensor& low,
                         const Tensor& width, const Tensor& prefix, const Tensor& keys,
                         const Tensor& values, const Tensor& flag, int64_t offset,
                         int64_t count, int64_t c, int64_t ny, int64_t nz, int64_t nf, bool shared) {
    check_table(keys,values,flag);
    for (const auto& t : {base,cells,low,width,prefix}) check(t,values,at::kLong);
    for (const auto& t : {phi,shape,weights}) check(t,values,values.scalar_type());
    const int64_t lattices=shared?2:3;
    TORCH_CHECK(base.dim()==3 && base.size(0)==lattices && base.size(2)==3, "IB base shape mismatch");
    TORCH_CHECK(phi.dim()==4 && phi.size(0)==lattices && phi.size(1)==base.size(1)
                && phi.size(2)==3 && phi.size(3)==4, "IB weight table shape mismatch");
    TORCH_CHECK(shape.dim()==2 && shape.size(1)==4 && shape.size(0)>0, "P1 shape values must be Q x 4");
    TORCH_CHECK(cells.dim()==2 && cells.size(1)==4, "P1 cells must be E x 4");
    const auto e=cells.size(0), q=shape.size(0);
    TORCH_CHECK(weights.dim()==2 && weights.size(0)==e && weights.size(1)==q, "IB quadrature shape mismatch");
    TORCH_CHECK(low.dim()==2 && low.size(0)==e && low.size(1)==3 && width.sizes()==low.sizes(), "IB bounds shape mismatch");
    TORCH_CHECK(prefix.dim()==1 && prefix.numel()==e+1, "IB prefix shape mismatch");
    TORCH_CHECK(offset>=0 && offset<=base.size(1) && weights.numel()<=base.size(1)-offset,
                "IB group offset is out of range");
    TORCH_CHECK(count>=0 && (count==0 || e>0) && c>=0 && c<3, "invalid IB site count or component");
    TORCH_CHECK(ny>0 && nz>0 && nf>0 && ny<=std::numeric_limits<int64_t>::max()/nz
                && nf%(ny*nz)==0, "invalid IB face dimensions");
}

static void hash_contract(const Tensor& base,const Tensor& phi,const Tensor& shape,
    const Tensor& weights,const Tensor& cells,const Tensor& low,const Tensor& width,
    const Tensor& prefix,const Tensor& keys,const Tensor& values,const Tensor& flag,
    int64_t offset,int64_t count,int64_t c,int64_t ny,int64_t nz,int64_t nf,bool shared) {
    check_inputs(base,phi,shape,weights,cells,low,width,prefix,keys,values,flag,offset,count,c,ny,nz,nf,shared);
    ib_contract_cuda(base,phi,shape,weights,cells,low,width,prefix,keys,values,flag,
                     Tensor(),Tensor(),Tensor(),offset,count,c,ny,nz,nf,shared,false);
}

static void cached_contract(const Tensor& base,const Tensor& phi,const Tensor& shape,
    const Tensor& weights,const Tensor& cells,const Tensor& low,const Tensor& width,
    const Tensor& prefix,const Tensor& keys,const Tensor& values,const Tensor& flag,
    const Tensor& missing_keys,const Tensor& missing_values,const Tensor& missing_count,
    int64_t offset,int64_t count,int64_t c,int64_t ny,int64_t nz,int64_t nf,bool shared) {
    check_inputs(base,phi,shape,weights,cells,low,width,prefix,keys,values,flag,offset,count,c,ny,nz,nf,shared);
    check(missing_keys,values,at::kLong);
    check(missing_values,values,values.scalar_type());
    check(missing_count,values,at::kLong);
    TORCH_CHECK(missing_keys.dim()==1 && missing_values.dim()==1
                && missing_keys.numel()==missing_values.numel() && missing_count.numel()==1,
                "invalid IB missing stream");
    ib_contract_cuda(base,phi,shape,weights,cells,low,width,prefix,keys,values,flag,
                     missing_keys,missing_values,missing_count,offset,count,c,ny,nz,nf,shared,true);
}

static void hash_accumulate(const Tensor& input_keys,const Tensor& input_values,
                            const Tensor& keys,const Tensor& values,const Tensor& flag) {
    check_table(keys,values,flag);
    check(input_keys,values,at::kLong);
    check(input_values,values,values.scalar_type());
    TORCH_CHECK(input_keys.dim()==1 && input_values.dim()==1
                && input_keys.numel()==input_values.numel(), "IB entry stream shape mismatch");
    ib_accumulate_cuda(input_keys,input_values,keys,values,flag);
}

TORCH_LIBRARY(afsi_ib_cuda,m) {
    m.def("hash_contract_(Tensor base, Tensor phi, Tensor shape, Tensor weights, Tensor cells, Tensor low, Tensor width, Tensor prefix, Tensor(a!) keys, Tensor(b!) values, Tensor(c!) flag, int offset, int count, int component, int ny, int nz, int nf, bool shared) -> ()");
    // Cached lookup reads immutable keys; insertions happen in a later launch.
    m.def("cached_contract_(Tensor base, Tensor phi, Tensor shape, Tensor weights, Tensor cells, Tensor low, Tensor width, Tensor prefix, Tensor keys, Tensor(a!) values, Tensor(b!) flag, Tensor(c!) missing_keys, Tensor(d!) missing_values, Tensor(e!) missing_count, int offset, int count, int component, int ny, int nz, int nf, bool shared) -> ()");
    m.def("hash_accumulate_(Tensor input_keys, Tensor input_values, Tensor(a!) keys, Tensor(b!) values, Tensor(c!) flag) -> ()");
}
TORCH_LIBRARY_IMPL(afsi_ib_cuda,CUDA,m) {
    m.impl("hash_contract_", &hash_contract);
    m.impl("cached_contract_", &cached_contract);
    m.impl("hash_accumulate_", &hash_accumulate);
}
