"""Retire cyclic graph owners before recapture, including checkpoint resume."""
from contextlib import contextmanager, nullcontext
import gc
from types import SimpleNamespace
import weakref
import pytest
import torch
from afsi_torch.mac.graph_capture import capture_initialization


class CyclicOwner:
    def __init__(self, value):
        self.value = value
        self.cycle = self


def retire(value, finalized, capturing):
    owner = CyclicOwner(value)
    ref = weakref.ref(owner)
    weakref.finalize(owner, lambda: finalized.append(capturing()))
    return ref


@pytest.mark.parametrize('enabled', [True, False])
@pytest.mark.parametrize('fail', [False, True])
def test_capture_initialization_restores_gc_and_defers_cleanup(enabled, fail):
    original = gc.isenabled()
    gc.disable()
    finalized = []
    old = retire(object(), finalized, lambda: False)
    if enabled:
        gc.enable()
    try:
        with pytest.raises(RuntimeError, match='capture failed') if fail else nullcontext():
            with capture_initialization():
                assert old() is None and finalized == [False]
                assert not gc.isenabled()
                pending = retire(object(), finalized, lambda: False)
                # Allocation pressure must not run cyclic finalizers in capture.
                garbage = [CyclicOwner(None) for _ in range(2000)]
                del garbage
                assert pending() is not None and finalized == [False]
                if fail:
                    raise RuntimeError('capture failed')
        assert gc.isenabled() == enabled
    finally:
        gc.collect()
        gc.enable() if original else gc.disable()
    assert pending() is None and finalized == [False, False]


@pytest.mark.parametrize('kind', ['mass', 'pressure3d', 'pressure2d'])
def test_all_capture_initializers_keep_cleanup_outside_capture(monkeypatch, kind):
    from afsi_torch.mac.mass_graph import GraphMassSolver
    from afsi_torch.mac.mg_workspace import MGWorkspace
    from afsi_torch.mac2d.mg_workspace import ChannelWorkspace
    cls = dict(mass=GraphMassSolver, pressure3d=MGWorkspace,
               pressure2d=ChannelWorkspace)[kind]
    solver = cls.__new__(cls)
    solver.options = SimpleNamespace(check_every=2)
    solver.graphs = {}
    sample = torch.zeros(4, dtype=torch.float64)
    solver.x, solver.r, solver.d, solver.Ad = [sample.clone() for _ in range(4)]
    solver.tol = sample.new_empty(())
    solver.p, solver.rhs = [sample.clone()], [sample.clone()]
    stream = SimpleNamespace(wait_stream=lambda other: None)
    capturing = [False]
    finalized, refs, counts = [], [], []

    @contextmanager
    def graph_scope(graph, stream):
        capturing[0] = True
        try:
            yield
        finally:
            capturing[0] = False

    def block(count):
        assert not gc.isenabled()
        if capturing[0]:
            counts.append(count)
            refs.append(retire(object(), finalized, lambda: capturing[0]))
            garbage = [CyclicOwner(None) for _ in range(2000)]
            del garbage

    solver._block = block
    monkeypatch.setattr(torch.cuda, 'device', lambda device: nullcontext())
    monkeypatch.setattr(torch.cuda, 'Stream', lambda **kwargs: stream)
    monkeypatch.setattr(torch.cuda, 'current_stream', lambda *args: stream)
    monkeypatch.setattr(torch.cuda, 'stream', lambda stream: nullcontext())
    monkeypatch.setattr(torch.cuda, 'CUDAGraph', object)
    monkeypatch.setattr(torch.cuda, 'graph', graph_scope)
    solver._capture()
    gc.collect()
    assert counts == [1, 2] and set(solver.graphs) == {1, 2}
    assert all(ref() is None for ref in refs) and finalized == [False, False]


@pytest.mark.skipif(not torch.cuda.is_available(), reason='CUDA unavailable')
@pytest.mark.parametrize('kind', ['mass', 'pressure3d', 'pressure2d'])
@torch.no_grad()
def test_rebuild_cuda_solver_with_cyclic_retired_graphs(kind):
    pytest.importorskip('triton')
    if kind == 'mass':
        from afsi_torch.mac.mass_graph import GraphMassSolver
        from afsi_torch.fluid.solvers import SolverOptions
        n = 24
        mass = (torch.eye(n, device='cuda', dtype=torch.float64)*3
                - torch.diag(torch.ones(n-1, device='cuda', dtype=torch.float64), 1)
                - torch.diag(torch.ones(n-1, device='cuda', dtype=torch.float64), -1)).to_sparse_csr()
        diagonal = torch.full((n, 3), 3., device='cuda', dtype=torch.float64)
        expected = torch.sin(torch.arange(n*3, device='cuda', dtype=torch.float64).reshape(n, 3))
        rhs = mass @ expected
        def create():
            return GraphMassSolver(mass, diagonal, SolverOptions(check_every=4))
    elif kind == 'pressure3d':
        from afsi_torch.mac import MACGrid, GeometricMultigrid
        from afsi_torch.mac.grid import negative_laplacian
        grid = MACGrid((8, 8, 8), (2., 3., 4.))
        expected = torch.cos(grid.coordinates(device='cuda').sum(-1))
        expected -= expected.mean()
        rhs = negative_laplacian(expected, grid.spacing)
        def create():
            return GeometricMultigrid(grid, device='cuda', backend='graph')
    else:
        from afsi_torch.mac2d.grid import ChannelGrid, negative_laplacian
        from afsi_torch.mac2d.multigrid import ChannelMultigrid
        grid = ChannelGrid((32, 8))
        expected = torch.cos(grid.coordinates(device='cuda').sum(-1))
        rhs = negative_laplacian(expected, grid.spacing)
        def create():
            return ChannelMultigrid(grid, device='cuda', backend='graph')

    original, thresholds = gc.isenabled(), gc.get_threshold()
    finalized = []
    try:
        for _ in range(3):
            solver = create()
            result, info = solver.solve(rhs)
            tolerance = info.tolerance if kind == 'mass' else info['tolerance']
            residual = info.residual_norm if kind == 'mass' else info['residual_norm']
            assert residual <= tolerance
            torch.testing.assert_close(result, expected, rtol=2e-8, atol=2e-9)
            gc.disable()
            old = retire(solver, finalized, torch.cuda.is_current_stream_capturing)
            del solver
            gc.set_threshold(20, 1, 1)
            # The next initializer must retire old graphs before any capture.
            gc.enable()
        solver = create()
        assert old() is None
    finally:
        gc.set_threshold(*thresholds)
        gc.collect()
        gc.enable() if original else gc.disable()
    assert finalized == [False]*3
