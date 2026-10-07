// Forward IB transfer for P2 quadrature and nodal FEM lattices.
#include <ATen/ATen.h>
#include <torch/library.h>
#include <cmath>
using at::Tensor;

Tensor ib_indexed_cuda(const Tensor&, const Tensor&, const Tensor&, int64_t, double, bool);
Tensor ib_compact_cuda(const Tensor&, const Tensor&, const Tensor&, int64_t, int64_t, int64_t, double, bool);

static void same(const Tensor& t, const Tensor& f, at::ScalarType dtype) {
    TORCH_CHECK(t.is_cuda() && t.device()==f.device(), "IB transfer requires one CUDA device");
    TORCH_CHECK(t.is_contiguous() && t.scalar_type()==dtype, "IB transfer requires contiguous matching tensors");
    TORCH_CHECK(!t.requires_grad(), "IB forward transfer does not register autograd");
}
static void field(const Tensor& f) {
    TORCH_CHECK(f.scalar_type()==at::kFloat || f.scalar_type()==at::kDouble, "IB supports float32/float64");
    same(f,f,f.scalar_type());
}
static Tensor indexed(const Tensor& f,const Tensor& ids,const Tensor& w,int64_t size,double volume,bool spread) {
    field(f); same(ids,f,at::kLong); same(w,f,f.scalar_type());
    TORCH_CHECK(f.dim()==2 && f.size(1)>0, "IB indexed field must be N x C");
    TORCH_CHECK(ids.dim()==2 && ids.size(1)>0 && ids.sizes()==w.sizes(), "IB indexed stencil shape mismatch");
    TORCH_CHECK(size>0 && std::isfinite(volume) && volume>0, "invalid IB output size/volume");
    TORCH_CHECK(!spread || f.size(0)==ids.size(0), "IB spread point count mismatch");
    TORCH_CHECK(spread || f.size(0)>0, "empty IB grid");
    return ib_indexed_cuda(f,ids,w,size,volume,spread);
}
static Tensor gather(const Tensor& f,const Tensor& ids,const Tensor& w) {
    return indexed(f,ids,w,1,1.,false);
}
static Tensor spread(const Tensor& f,const Tensor& ids,const Tensor& w,int64_t size,double volume) {
    return indexed(f,ids,w,size,volume,true);
}
static Tensor compact(const Tensor& f,const Tensor& base,const Tensor& phi,int64_t nx,int64_t ny,int64_t nz,double volume,bool spread) {
    field(f); same(base,f,at::kLong); same(phi,f,f.scalar_type());
    TORCH_CHECK(f.dim()==1 && base.dim()==2 && base.size(1)==3, "invalid compact IB field/base");
    TORCH_CHECK(phi.dim()==3 && phi.size(0)==base.size(0) && phi.size(1)==3 && phi.size(2)==4, "invalid compact IB weights");
    TORCH_CHECK(nx>0 && ny>0 && nz>0 && std::isfinite(volume) && volume>0, "invalid compact grid/volume");
    TORCH_CHECK(f.numel()==(spread?base.size(0):nx*ny*nz), "compact IB field length mismatch");
    return ib_compact_cuda(f,base,phi,nx,ny,nz,volume,spread);
}
static Tensor cgather(const Tensor& f,const Tensor& b,const Tensor& p,int64_t nx,int64_t ny,int64_t nz) {
    return compact(f,b,p,nx,ny,nz,1.,false);
}
static Tensor cspread(const Tensor& f,const Tensor& b,const Tensor& p,int64_t nx,int64_t ny,int64_t nz,double volume) {
    return compact(f,b,p,nx,ny,nz,volume,true);
}
TORCH_LIBRARY_FRAGMENT(afsi_ib_cuda,m) {
    m.def("indexed_gather(Tensor field, Tensor indices, Tensor weights) -> Tensor");
    m.def("indexed_spread(Tensor field, Tensor indices, Tensor weights, int size, float volume) -> Tensor");
    m.def("compact_gather(Tensor field, Tensor base, Tensor phi, int nx, int ny, int nz) -> Tensor");
    m.def("compact_spread(Tensor field, Tensor base, Tensor phi, int nx, int ny, int nz, float volume) -> Tensor");
}
TORCH_LIBRARY_IMPL(afsi_ib_cuda,CUDA,m) {
    m.impl("indexed_gather",&gather); m.impl("indexed_spread",&spread);
    m.impl("compact_gather",&cgather); m.impl("compact_spread",&cspread);
}
