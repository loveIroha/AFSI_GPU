"""Structural interfaces; coupled solvers do not select constitutive laws."""
from typing import Protocol, Any
import torch


class CSRTangent(Protocol):
    """Derivative of integrated nodal force, not stress or residual stiffness."""
    def assemble(self,x:torch.Tensor,time:float)->torch.Tensor: ...
    def diagonal(self,tangent:torch.Tensor)->torch.Tensor: ...


class SolidModel(Protocol):
    mesh: Any                 # reference X=(nodes,3), cells=(elements,local_nodes)
    geometry: Any             # prepared reference finite-element geometry
    def force(self,x:torch.Tensor,time:float)->torch.Tensor: ...
    def validate(self,x:torch.Tensor)->None: ...
    def tangent_factory(self,chunk_size:int=2048)->CSRTangent: ...


class SolidExecution(Protocol):
    """Optional checked geometry/force cache used by optimized coupling."""
    model: SolidModel
    def force(self,x:torch.Tensor,time:float)->torch.Tensor: ...
    def validate(self,x:torch.Tensor)->None: ...
    def force_with_geometry(self,x:torch.Tensor,time:float)->tuple: ...
    def remember_geometry(self,x:torch.Tensor,geometry:tuple)->None: ...


def require_tangent(model):
    if not callable(getattr(model,'tangent_factory',None)):
        raise TypeError('implicit elasticity requires solid.tangent_factory(chunk_size), returning an assembled CSR force derivative')


def make_tangent(model,chunk_size=2048):
    require_tangent(model)
    tangent=model.tangent_factory(chunk_size)
    if not all(callable(getattr(tangent,name,None)) for name in ('assemble','diagonal')):
        raise TypeError('solid tangent must provide assemble(x,time) and diagonal(CSR)')
    return tangent


def make_execution(model,backend,*,optimized=False):
    name='pointwise_execution_factory' if backend=='pointwise' else 'execution_factory'
    factory=getattr(model,name,None)
    if not callable(factory):
        raise TypeError(f'fused solid execution requires solid.{name}(); use execution_backend="torch" for reference callbacks')
    execution=factory()
    names=('force','validate','force_with_geometry','remember_geometry') if optimized else ('force','validate')
    if not all(callable(getattr(execution,n,None)) for n in names):
        raise TypeError('solid execution does not implement the requested coupling capabilities')
    if getattr(execution,'model',None) is not model:
        raise TypeError('solid execution must retain its source model')
    return execution


def failure_diagnostics(model,grid,state,problem):
    """Optional model-local detail; no material assumptions in the integrator."""
    callback=getattr(model,'failure_diagnostics',None)
    if callable(callback):
        return callback(grid,state,problem)
    report=dict(accepted_step=state.step,accepted_time_s=state.time,
        accepted_velocity_component_max_cm_per_s=[u.abs().max().item() for u in state.velocity],
        residual_evaluations=problem.evaluations,jacobian_actions=problem.actions,
        tangent_assemblies=problem.assemblies,
        trial_status='last residual evaluation; not an accepted checkpoint')
    diagnostics=getattr(model,'diagnostics',None)
    if callable(diagnostics):
        report['accepted_solid']=diagnostics(state.x)
    return report
