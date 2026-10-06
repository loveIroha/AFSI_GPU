"""Cell-resident numeric contraction: padding, layouts, exact dual operators."""
from dataclasses import replace
from math import prod
from types import SimpleNamespace
import pytest
import torch
from afsi_torch.mac.assembled_transfer import AssembledP1Transfer, _cpu_entries, _bases
from afsi_torch.mac.adaptive_transfer import QuadratureGroup
from test_gpu_coupled_work import pair, DEVICES


def cell_kernel_case(device,dtype,q,layout,component):
    """Different cell widths, nonzero group offset and >int32 encoded keys."""
    E,offset = 3,5
    p = offset+E*q+7
    groups = 2 if layout=='shared' else 3
    rng = torch.Generator().manual_seed(661)
    # Small integer base variation gives unequal cells and partial final tiles.
    base = torch.randint(0,3,(groups,p,3),generator=rng,dtype=torch.int64)
    base[:,offset:offset+q] = 0
    base += torch.tensor([17,28,39])
    phi = torch.rand((groups,p,3,4),generator=rng,dtype=dtype)
    shape = torch.rand((q,4),generator=rng,dtype=dtype)
    shape /= shape.sum(-1,keepdim=True)
    weights = .01+torch.rand((E,q),generator=rng,dtype=dtype)
    cells = torch.tensor([[1700,1701,1702,1703],[1701,1704,1703,1705],
                          [1703,1702,1705,1706]],dtype=torch.int64)
    stencil = SimpleNamespace(base=base.to(device),phi=phi.to(device),layout=layout)
    group = QuadratureGroup(cells.to(device),shape.to(device),weights.to(device))
    b = _bases(stencil,component,offset,E*q).reshape(E,q,3)
    low = b.amin(1)
    width = b.amax(1)-low+4
    prefix = torch.cat((width.new_zeros(1),width.prod(-1).cumsum(0)))
    return stencil,group,offset,low,width,prefix,int(prefix[-1]),(128,129,130)


def check_actual_cell_entries(device,dtype,q,layout,component):
    import triton
    from afsi_torch.mac._triton_cell_transfer_assembly import _cell_entries
    s,g,offset,low,width,prefix,count,face = cell_kernel_case(device,dtype,q,layout,component)
    # Guards on both sides detect padded-lane writes, rather than allowing
    # ignored entries in the final tile to hide corruption.
    keys = torch.full((count*4+8,),-123,device=device,dtype=torch.int64)
    values = torch.full((count*4+8,),-987.,device=device,dtype=dtype)
    B = 8 if q<=128 else 4
    _cell_entries[(len(g.cells),)](s.base,s.phi,g.values,g.weights,g.cells,low,width,prefix,
        keys[4:],values[4:],keys,s.base.shape[1],q,component,layout=='shared',offset,
        face[1],face[2],prod(face),B,triton.next_power_of_2(q),False,0,False,
        keys,values,keys,0,num_warps=4 if q<=128 else 8,enable_fp_fusion=False)
    expected_keys,expected_values = _cpu_entries(s,g,component,offset,low,width,prefix,0,count,face)
    torch.testing.assert_close(keys[4:-4],expected_keys,atol=0,rtol=0)
    tolerance = dict(rtol=3e-5,atol=3e-6) if dtype==torch.float32 else dict(rtol=3e-11,atol=3e-12)
    torch.testing.assert_close(values[4:-4],expected_values,**tolerance)
    assert expected_keys.max()>2147483647
    assert torch.all(keys[:4]==-123) and torch.all(keys[-4:]==-123)
    assert torch.all(values[:4]==-987.) and torch.all(values[-4:]==-987.)


@pytest.mark.skipif(not torch.cuda.is_available(),reason='CUDA unavailable')
@pytest.mark.parametrize('dtype',[torch.float32,torch.float64])
@pytest.mark.parametrize('q,layout,component',[(3,'shared',0),(21,'component',1),
                                            (129,'shared',2),(257,'component',0)])
def test_actual_cell_kernel_padding_offsets_layouts_and_large_keys(dtype,q,layout,component):
    pytest.importorskip('triton')
    with torch.cuda.device('cuda'):
        check_actual_cell_entries('cuda',dtype,q,layout,component)


@pytest.mark.parametrize('device',DEVICES)
@pytest.mark.parametrize('layout',['shared','component'])
@pytest.mark.parametrize('backend',['hash','cached-hash'])
def test_cell_csr_changed_support_rule_and_older_snapshots_keep_duality(device,layout,backend):
    X,reference,oracle = pair(device,torch.float64,layout)
    cell = AssembledP1Transfer(reference.grid,reference.geometry,
        quadrature_options=reference.quadrature_options,options=reference.options,
        fused=False,chunk_entries=16384,assembly_backend=backend,contraction_backend='cell')
    first = cell.prepare(X)
    saved = [B.values().clone() for B in first.gather+first.spread]
    # Rescale, translate and repeat: different adaptive rules, missing support,
    # and the steady numeric-only path all use the new contraction dispatch.
    moved = (X-4)*1.4+X.new_tensor([4.3,4.2,4.1])
    second = cell.prepare(moved)
    third = cell.prepare(moved)
    assert not torch.equal(first.rule.orders,second.rule.orders)
    assert second.assembly['contraction_backend']=='cell'
    assert second.assembly['contraction_execution']==('triton-cell-resident' if device=='cuda' else 'torch-cpu-oracle')
    for result in (second,third):
        expected = oracle.prepare(moved)
        for actual,B in zip(result.gather+result.spread,expected.gather+expected.spread):
            torch.testing.assert_close(actual.to_dense(),B.to_dense(),rtol=3e-11,atol=3e-12)
        for B,BT in zip(result.gather,result.spread):
            torch.testing.assert_close(BT.to_dense(),B.to_dense().T,rtol=0,atol=0)
    for B,value in zip(first.gather+first.spread,saved):
        torch.testing.assert_close(B.values(),value,rtol=0,atol=0)
    force = torch.sin(moved)
    velocity = tuple(torch.sin(reference.grid.coordinates(c,device=device)[...,c]) for c in range(3))
    density,_ = cell.spread(force,third)
    nodal,_ = cell.interpolate(velocity,third)
    torch.testing.assert_close((force*nodal).sum(),
        reference.grid.volume*sum((u*f).sum() for u,f in zip(velocity,density)),rtol=3e-11,atol=3e-12)
    if backend=='cached-hash':
        assert all(i['symbolic_reused'] for i in third.assembly['components'])


def test_cell_cli_config_and_unsupported_combinations(tmp_path):
    from demo.real_lv_fsi.run_mac import main
    from afsi_torch.config import load_config
    from afsi_torch.paper_lv import PaperLVConfig
    path = tmp_path/'config.json'
    main(['--ib-response-backend','csr','--ib-csr-assembly-backend','cached-hash',
          '--ib-csr-contraction-backend','cell','--write-config',str(path)])
    cfg = load_config(path,PaperLVConfig)
    assert cfg.ib_csr_contraction_backend=='cell'
    assert cfg.material==PaperLVConfig().material and cfg.time==PaperLVConfig().time
    with pytest.raises(ValueError,match='cell contraction requires'):
        replace(cfg,ib_csr_assembly_backend='coalesce')
    with pytest.raises(ValueError,match='cell contraction requires'):
        replace(cfg,ib_response_backend='quadrature')
    with pytest.raises(ValueError,match='contraction_backend'):
        replace(cfg,ib_csr_contraction_backend='invalid')


@pytest.mark.parametrize('device',DEVICES)
def test_cell_missing_stream_overflow_rebuild_keeps_current_matrix_and_old_values(device):
    X,r,oracle = pair(device,torch.float64)
    transfer = AssembledP1Transfer(r.grid,r.geometry,quadrature_options=r.quadrature_options,
        options=r.options,fused=False,chunk_entries=4,assembly_backend='cached-hash',contraction_backend='cell')
    old = transfer.prepare(X)
    copies = [B.values().clone() for B in old.gather+old.spread]
    moved = X+X.new_tensor([.9,.9,.9])
    current = transfer.prepare(moved)
    assert any(i['symbolic_reason']=='missing-stream-overflow-reset' for i in current.assembly['components'])
    expected = oracle.prepare(moved)
    for a,b in zip(current.gather+current.spread,expected.gather+expected.spread):
        torch.testing.assert_close(a.to_dense(),b.to_dense(),rtol=3e-11,atol=3e-12)
    for B,values in zip(old.gather+old.spread,copies):
        torch.testing.assert_close(B.values(),values,atol=0,rtol=0)


def test_cell_benchmark_profile_continues_cache_and_checkpoint_override(tmp_path):
    # A small open well with a positive cavity: no Gmsh or external mesh files.
    from itertools import product,permutations
    from afsi_torch.mesh_io import ImportedSolidMesh
    from afsi_torch.paper_lv import PaperLVConfig,PaperLVSolid
    from afsi_torch.config import TimeConfig,FluidConfig,OutputConfig,LVExecutionConfig
    from afsi_torch.mac.adaptive_transfer import InteractionQuadratureOptions
    from afsi_torch.mac.paper_coupling import BEIBStepper
    from afsi_torch.paper_lv_checkpoint import save,load
    from afsi_torch.simulation.paper_lv_mac import run
    from afsi_torch.nonlinear import coupled_linear_policy
    from validation.benchmark_paper_lv import benchmark
    xyz = list(product(range(4),range(4),range(3)))
    ids = {p:i for i,p in enumerate(xyz)}
    X = torch.tensor(xyz,dtype=torch.float64)+5
    elements = []
    for origin in product(range(3),range(3),range(2)):
        if origin==(1,1,1):
            continue
        for order in permutations(range(3)):
            vertex = list(origin)
            tetra = [ids[tuple(vertex)]]
            for axis in order:
                vertex[axis] += 1
                tetra.append(ids[tuple(vertex)])
            if torch.linalg.det((X[tetra[1:]]-X[tetra[0]]).T)<0:
                tetra[1],tetra[2] = tetra[2],tetra[1]
            elements.append(tetra)
    cells = torch.tensor(elements)
    owners = {}
    local_faces = ((1,2,3),(0,3,2),(0,1,3),(0,2,1))
    for e,cell in enumerate(elements):
        for local,vertices in enumerate(local_faces):
            face = [cell[i] for i in vertices]
            owners.setdefault(tuple(sorted(face)),[]).append((face,e,local))
    faces,bc,bl,tags = [],[],[],[]
    for entries in owners.values():
        if len(entries)!=1:
            continue
        face,e,local = entries[0]
        center = X[face].mean(0)
        tag = 1 if 6<=center[0]<=7 and 6<=center[1]<=7 and center[2]>=6 else 2
        if center[2]==7:
            tag = 3
        faces.append(face); bc.append(e); bl.append(local); tags.append(tag)
    mesh = ImportedSolidMesh(X,cells,torch.tensor(faces),torch.tensor(tags),torch.tensor(bc),
        torch.tensor(bl),X.new_tensor([[1.,0.,0.]]).expand(len(cells),3).clone(),
        X.new_tensor([[0.,1.,0.]]).expand(len(cells),3).clone(),len(X),{'units':'cm'})
    cfg = PaperLVConfig(time=TimeConfig(1e-4,2e-4),fluid=FluidConfig((8,)*3,(13.,)*3,mu=.04),
        execution=LVExecutionConfig(warm_start=True),output=OutputConfig(1,1,1,False),
        interaction_quadrature=InteractionQuadratureOptions(mode='adaptive',stencil_backend='shared'),
        nonlinear_solver='anderson-newton',ib_response_backend='csr',ib_csr_assembly_backend='cached-hash')
    cfg = replace(cfg,nonlinear=coupled_linear_policy(cfg.nonlinear,'inexact'))
    model = PaperLVSolid(mesh,cfg)
    path = tmp_path/'run'/'checkpoint.npz'
    save(path,model,BEIBStepper(model,cfg,'cpu').initialize(X),cfg,{'elapsed_seconds':0.})
    original = path.read_bytes()
    report = benchmark(path,device='cpu',warmup=1,steps=1,intervals=(5,),solvers=('anderson-newton',),
        csr_contraction_backends=('sites','cell'),profile=True,profile_steps=1)
    assert path.read_bytes()==original
    assert all(v['status']=='completed' and v['max_accepted_residual_to_tolerance']<=1 for v in report['variants'])
    for v in report['variants']:
        assert v['profile']['start_time_s']==v['end_time_s']
        assert v['profile']['end_time_s']==pytest.approx(v['end_time_s']+cfg.time.dt)
        assert v['profile']['ib_csr_cache_per_step']['rebuilds']>=0
    run(resume=path,device='cpu',end_time=cfg.time.dt,ib_csr_contraction_backend='cell')
    _,state,restored,_ = load(path)
    assert state.step==1 and restored.ib_csr_contraction_backend=='cell'
    assert restored.material==cfg.material and restored.time.dt==cfg.time.dt
