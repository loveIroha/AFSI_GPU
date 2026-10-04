"""Replaceable finite-element solid models for shared MAC/IB integration."""
from .contracts import SolidModel, CSRTangent, SolidExecution, make_tangent
from .p1_model import P1Solid, BoundaryForce

__all__=['SolidModel','CSRTangent','SolidExecution','make_tangent','P1Solid','BoundaryForce']
