// Forward P2 quadrature transfer. The consistent mass solve remains in PyTorch.
#include <ATen/ATen.h>
#include <torch/library.h>
#include <cmath>
using at::Tensor;

void p2_prepare_cuda(const Tensor&,const Tensor&,const Tensor&,const Tensor&,const Tensor&,
    const Tensor&,const Tensor&,const Tensor&,const Tensor&,int64_t);
void p2_transfer_cuda(const Tensor&,const Tensor&,const Tensor&,const Tensor&,const Tensor&,
    const Tensor&,const Tensor&,const Tensor&,const Tensor&,const Tensor&,
    int64_t,int64_t,int64_t,int64_t,double,bool);

static void check(const Tensor& t,const Tensor& like,at::ScalarType dtype) {
    TORCH_CHECK(t.is_cuda() && t.device()==like.device() && t.is_contiguous(),
        "P2 IB requires contiguous tensors on one CUDA device");
    TORCH_CHECK(t.scalar_type()==dtype && !t.requires_grad(),"P2 IB dtype/autograd mismatch");
}
static void field(const Tensor& t) {
    TORCH_CHECK(t.scalar_type()==at::kDouble || t.scalar_type()==at::kFloat,"P2 IB requires FP32/FP64");
    check(t,t,t.scalar_type());
}
static void group(const Tensor& cells,const Tensor& N,const Tensor& like) {
    check(cells,like,at::kLong); check(N,like,like.scalar_type());
    TORCH_CHECK(cells.dim()==2 && cells.size(1)==10 && N.dim()==2 && N.size(1)==10 && N.size(0)>0,
        "P2 transfer requires ten basis functions");
}
static void tables(const Tensor& base,const Tensor& phi,const Tensor& like,int64_t count,int64_t offset) {
    check(base,like,at::kLong); check(phi,like,like.scalar_type());
    TORCH_CHECK(base.dim()==3 && base.size(0)==2 && base.size(2)==3 &&
        phi.dim()==4 && phi.size(0)==2 && phi.size(1)==base.size(1) && phi.size(2)==3 && phi.size(3)==4,
        "P2 IB requires shared face/center tables");
    TORCH_CHECK(offset>=0 && count>=0 && offset<=base.size(1) && count<=base.size(1)-offset,
        "P2 IB group exceeds table bounds");
}
static void prepare(const Tensor& x,const Tensor& cells,const Tensor& N,const Tensor& origin,
    const Tensor& spacing,const Tensor& limits,const Tensor& base,const Tensor& phi,
    const Tensor& invalid,int64_t offset) {
    field(x);group(cells,N,x);tables(base,phi,x,cells.size(0)*N.size(0),offset);
    check(origin,x,x.scalar_type());check(spacing,x,x.scalar_type());check(limits,x,x.scalar_type());
    check(invalid,x,at::kInt);
    TORCH_CHECK(x.dim()==2 && x.size(1)==3 && origin.numel()==3 && spacing.numel()==3 &&
        limits.numel()==3 && invalid.numel()==1,"invalid P2 preparation inputs");
    p2_prepare_cuda(x,cells,N,origin,spacing,limits,base,phi,invalid,offset);
}
static void transfer(const Tensor& value,const Tensor& u0,const Tensor& u1,const Tensor& u2,
    const Tensor& cells,const Tensor& N,const Tensor& W,const Tensor& base,const Tensor& phi,
    const Tensor& rhs,int64_t offset,int64_t nx,int64_t ny,int64_t nz,double volume,bool spread) {
    field(value);group(cells,N,value);tables(base,phi,value,cells.size(0)*N.size(0),offset);
    for (auto t:{u0,u1,u2,W,rhs})check(t,value,value.scalar_type());
    TORCH_CHECK(value.dim()==2 && value.size(1)==3 && rhs.sizes()==value.sizes(),"invalid P2 nodal field");
    TORCH_CHECK(W.dim()==2 && W.size(0)==cells.size(0) && W.size(1)==N.size(0),"P2 integration weights mismatch");
    TORCH_CHECK(nx>0 && ny>0 && nz>0 && std::isfinite(volume) && volume>0,"invalid P2 MAC grid");
    TORCH_CHECK(u0.numel()==(nx+1)*ny*nz && u1.numel()==nx*(ny+1)*nz &&
        u2.numel()==nx*ny*(nz+1),"P2 MAC field length mismatch");
    p2_transfer_cuda(value,u0,u1,u2,cells,N,W,base,phi,rhs,offset,nx,ny,nz,volume,spread);
}
static void gather(const Tensor& u0,const Tensor& u1,const Tensor& u2,const Tensor& cells,
    const Tensor& N,const Tensor& W,const Tensor& base,const Tensor& phi,const Tensor& rhs,
    int64_t offset,int64_t nx,int64_t ny,int64_t nz) {
    transfer(rhs,u0,u1,u2,cells,N,W,base,phi,rhs,offset,nx,ny,nz,1.,false);
}
static void spread(const Tensor& value,const Tensor& cells,const Tensor& N,const Tensor& W,
    const Tensor& base,const Tensor& phi,const Tensor& u0,const Tensor& u1,const Tensor& u2,
    int64_t offset,int64_t nx,int64_t ny,int64_t nz,double volume) {
    transfer(value,u0,u1,u2,cells,N,W,base,phi,value,offset,nx,ny,nz,volume,true);
}
TORCH_LIBRARY_FRAGMENT(afsi_ib_cuda,m) {
    m.def("p2_prepare_(Tensor x, Tensor cells, Tensor N, Tensor origin, Tensor spacing, Tensor limits, Tensor(a!) base, Tensor(b!) phi, Tensor(c!) invalid, int offset) -> ()");
    m.def("p2_gather_assemble_(Tensor u0, Tensor u1, Tensor u2, Tensor cells, Tensor N, Tensor W, Tensor base, Tensor phi, Tensor(a!) rhs, int offset, int nx, int ny, int nz) -> ()");
    m.def("p2_spread_(Tensor value, Tensor cells, Tensor N, Tensor W, Tensor base, Tensor phi, Tensor(a!) u0, Tensor(b!) u1, Tensor(c!) u2, int offset, int nx, int ny, int nz, float volume) -> ()");
}
TORCH_LIBRARY_IMPL(afsi_ib_cuda,CUDA,m) {
    m.impl("p2_prepare_",&prepare);m.impl("p2_gather_assemble_",&gather);m.impl("p2_spread_",&spread);
}
