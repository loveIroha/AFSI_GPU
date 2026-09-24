"""Synthetic ellipsoidal helix field with an explicit smooth apical escape.

Away from the apical core, fiber lies in the local ellipsoid tangent plane.
Inside the core it tilts out of plane to avoid the circumferential singularity.
This is a numerical rule-based field, not a histological reconstruction.
"""
from dataclasses import dataclass
from math import isfinite, pi
import torch


@dataclass(frozen=True)
class FiberField:
    fiber: torch.Tensor
    sheet: torch.Tensor
    transmural: torch.Tensor
    helix_angle: torch.Tensor  # radians, target angle outside the apical core
    apical_weight: torch.Tensor


def rule_based_fibers(X, config, *, endo_angle=60., epi_angle=-60., apical_width=.15):
    """Return nodal unit/orthogonal fiber and sheet, all on X.device.

    Transmural coordinate uses radial intersections with both ellipsoids;
    clamp for small inward chord errors of the faceted reference boundary.
    Angles in degrees. apical_width is dimensionless transverse normal length.
    Sheet is a smooth orthogonal completion, not a fitted anatomical sheet.
    """
    if X.ndim != 2 or X.shape[1] != 3 or not X.is_floating_point() or not torch.isfinite(X).all():
        raise ValueError('expected finite floating coordinates (N,3)')
    if not all(isfinite(v) for v in (endo_angle, epi_angle, apical_width)) or not 0 < apical_width < 1:
        raise ValueError('angles must be finite and apical_width must lie in (0,1)')
    y = X-X.new_tensor(config.center)
    radius = torch.linalg.vector_norm(y, dim=-1)
    if (radius <= torch.finfo(X.dtype).eps*max(config.outer_axes)).any():
        raise ValueError('fiber frame undefined at ellipsoid center (outside wall domain)')
    direction = y/radius[:, None]
    ai, ao = X.new_tensor(config.inner_axes), X.new_tensor(config.outer_axes)
    ri = 1/torch.linalg.vector_norm(direction/ai, dim=-1)
    ro = 1/torch.linalg.vector_norm(direction/ao, dim=-1)
    t = ((radius-ri)/(ro-ri)).clamp(0, 1)
    axes = ai+t[:, None]*(ao-ai)
    radial = y/axes.square()
    radial = radial/torch.linalg.vector_norm(radial, dim=-1, keepdim=True)
    # A smooth frame based at the SOUTH pole. The removed north pole is outside
    # the supported wall; fail explicitly if a caller supplies that point.
    denominator = 1-radial[:, 2]
    if (denominator < 1e-8).any():
        raise ValueError('north-pole frame singularity; expected truncated LV wall')
    nx, ny = radial[:, 0], radial[:, 1]
    e1 = torch.stack((1-nx.square()/denominator, -nx*ny/denominator, nx), -1)
    ez = torch.zeros_like(radial)
    ez[:, 2] = 1
    circum = torch.linalg.cross(ez, radial)
    longitudinal = torch.linalg.cross(radial, circum)
    transverse2 = circum.square().sum(-1)
    escape = (1-transverse2/apical_width**2).clamp(min=0).square()
    angle = ((1-t)*endo_angle+t*epi_angle)*(pi/180)
    f = angle.cos()[:, None]*circum+angle.sin()[:, None]*longitudinal+escape[:, None]*radial
    f = f/torch.linalg.vector_norm(f, dim=-1, keepdim=True)
    # Minimal rotation taking radial -> fiber also rotates e1 -> sheet.
    v = torch.linalg.cross(radial, f)
    cosine = (radial*f).sum(-1)
    sheet = e1+torch.linalg.cross(v, e1)+torch.linalg.cross(v, torch.linalg.cross(v, e1))/(1+cosine[:, None])
    sheet = sheet/torch.linalg.vector_norm(sheet, dim=-1, keepdim=True)
    return FiberField(f, sheet, t, angle, escape)
