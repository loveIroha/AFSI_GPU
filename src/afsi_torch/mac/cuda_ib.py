"""Optional PyTorch C++/CUDA operators for frozen P1 IB contraction.

Compilation is lazy; importing afsi_torch never requires a CUDA toolkit.
Tensor data stay on the GPU, and kernels use PyTorch's current CUDA stream.
The extension replaces numeric contraction/insertion only. CSR structure,
consistent mass solves and paired interpolation/spreading remain unchanged.
"""
from pathlib import Path
import os
import threading
import torch

_lock = threading.Lock()
_loaded = False


def build(*, verbose=False):
    """Build once in torch's extension cache; call before CUDA Graph capture."""
    global _loaded
    if _loaded:
        return
    if not torch.cuda.is_available() or torch.version.cuda is None:
        raise RuntimeError('IB cuda contraction requires a CUDA-enabled PyTorch and an available GPU')
    if torch.cuda.is_current_stream_capturing():
        raise RuntimeError('build the IB CUDA extension before CUDA Graph capture')
    with _lock:
        if _loaded:
            return
        from torch.utils.cpp_extension import CUDA_HOME, load
        if CUDA_HOME is None:
            raise RuntimeError('IB cuda contraction requires the CUDA toolkit (nvcc); set CUDA_HOME')
        source = Path(__file__).parent/'csrc'
        load(name='afsi_ib_cuda', sources=[str(source/'ib.cpp'), str(source/'ib.cu')],
             extra_cflags=['/O2'] if os.name=='nt' else ['-O3'],
             extra_cuda_cflags=['-O3', '--fmad=false', '-lineinfo'],
             is_python_module=False, verbose=verbose)
        _loaded = True


def contract(stencil, group, component, offset, low, width, prefix, count, face_shape,
             workspace, cache=None):
    """Integrate four FE contributions together; never materialize raw entries."""
    build()
    args = (stencil.base, stencil.phi, group.values, group.weights, group.cells,
            low, width, prefix, workspace.keys, workspace.values, workspace.flag)
    scalars = (offset, count, component, face_shape[1], face_shape[2],
               face_shape[0]*face_shape[1]*face_shape[2],
               getattr(stencil, 'layout', None)=='shared')
    if cache is None:
        torch.ops.afsi_ib_cuda.hash_contract_(*args, *scalars)
    else:
        torch.ops.afsi_ib_cuda.cached_contract_(*args, cache.missing_keys,
            cache.missing_values, cache.missing_count, *scalars)


def hash_accumulate(keys, values, workspace):
    build()
    torch.ops.afsi_ib_cuda.hash_accumulate_(keys, values, workspace.keys,
                                          workspace.values, workspace.flag)


if __name__ == '__main__':
    build(verbose=True)
    print('AFSI IB C++/CUDA operators ready')
