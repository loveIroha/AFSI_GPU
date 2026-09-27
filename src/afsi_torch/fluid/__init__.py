"""Cartesian Q2/Q1 fluid operators and forward-only Chorin solves."""
from .mesh import BoxMesh, create_box
from .operators import FluidOperators, prepare_operators
from .solvers import SolverOptions, pcg
from .chorin import ChorinSolver
from .csr import CSRFluidOperators

__all__ = ['BoxMesh', 'create_box', 'FluidOperators', 'prepare_operators',
           'SolverOptions', 'pcg', 'ChorinSolver', 'CSRFluidOperators']

