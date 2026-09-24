"""Compare exported actual DOLFINx assembly with PyTorch on CPU or CUDA.

No DOLFINx/Basix imports are required here. A mismatch raises and exits nonzero.
"""
import argparse
import hashlib
import json
from pathlib import Path
import numpy as np
import torch
from afsi_torch import boundary as bd, solid, materials
from afsi_torch.fields import prepare_reference_fields
try:
    from .reference_io import TERMS, load_reference
except ImportError:
    from reference_io import TERMS, load_reference


def select_tagged_faces(exterior, vertices, tags):
    """Carry exported facet tags by vertex IDs, never by reclassifying geometry."""
    lookup = {tuple(sorted(face[:3])): face for face in exterior}
    seen, selected = set(), {1: [], 2: []}
    for face, tag in zip(vertices, tags, strict=True):
        key = tuple(sorted(face))
        if key in seen or key not in lookup or int(tag) not in selected:
            raise ValueError("duplicate, non-exterior or unsupported tagged facet")
        selected[int(tag)].append(lookup[key])
        seen.add(key)
    if not all(selected.values()):
        raise ValueError("both pressure and spring regions must be nonempty")
    return {tag: np.stack(faces) for tag, faces in selected.items()}


def compare(path, device="cpu"):
    device = torch.device(device)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA requested but unavailable; no CPU fallback")
    data, meta = load_reference(path)
    tensor = lambda a: torch.as_tensor(a, dtype=torch.float64, device=device)
    X = tensor(data['X'])
    cells = torch.as_tensor(data['cells'], dtype=torch.int64, device=device)
    geo = solid.prepare_p2(X, cells, quadrature=(data['volume_points'], data['volume_weights']))
    fields = prepare_reference_fields(geo, data['fiber'], data['sheet'], data['tension'])
    parameters = materials.GuccioneParameters(**meta['parameters'])
    exterior = bd.extract_boundary(X, cells).cpu().numpy()
    regions = select_tagged_faces(exterior, data['tagged_vertices'], data['facet_tags'])
    surfaces = {tag: bd.prepare_surface(X, torch.as_tensor(faces, dtype=torch.int64, device=device),
        quadrature=(data['surface_points'], data['surface_weights'])) for tag, faces in regions.items()}
    pressure = bd.prepare_coefficient(surfaces[1], data['pressure'])
    beta = bd.prepare_coefficient(surfaces[2], data['beta'], nonnegative=True)

    def passive(x):
        F = solid.deformation_gradient(x, geo)
        return solid.assemble_pk1(materials.guccione_pk1(
            F, fields.fiber, fields.sheet, fields.normal, parameters), geo)

    def active(x):
        F = solid.deformation_gradient(x, geo)
        return solid.assemble_pk1(materials.active_pk1(F, fields.fiber, fields.tension), geo)

    kernels = dict(passive=passive, active=active,
                   pressure=lambda x: bd.pressure_force(x, surfaces[1], pressure),
                   spring=lambda x: bd.spring_force(x, surfaces[2], beta))
    parts = tuple(kernels.values())
    kernels['total'] = lambda x: sum(kernel(x) for kernel in parts)
    direction = tensor(data['direction'])
    reports = {}
    for index, name in enumerate(meta['cases']):
        x = tensor(data['x'][index])
        solid.validate_deformation(x, geo)
        for surface in surfaces.values():
            bd.validate_surface(x, surface)
        reports[name] = {}
        for term in TERMS:
            force, tangent = torch.func.jvp(kernels[term], (x,), (direction,))
            errors = {}
            for kind, actual, atol in (("force", force, 1e-8), ("tangent", tangent, 2e-7)):
                actual = actual.detach().cpu().numpy()
                expected = data[kind+'_'+term][index]
                np.testing.assert_allclose(actual, expected, rtol=3e-10, atol=atol,
                                           err_msg=f"{name}/{term}/{kind}")
                difference = actual-expected
                errors[kind+'_max_abs'] = float(np.abs(difference).max())
                errors[kind+'_scaled_l2'] = float(np.linalg.norm(difference)/max(1., np.linalg.norm(expected)))
            reports[name][term] = errors
    return dict(status="passed", device=str(device), torch=torch.__version__,
                reference_sha256=hashlib.sha256(Path(path).read_bytes()).hexdigest(),
                reference=meta, nodes=len(X), cells=len(cells), comparisons=reports)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--reference", default="validation/results/dolfinx_reference.npz")
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--output", help="Optional JSON validation report")
    args = parser.parse_args()
    report = json.dumps(compare(args.reference, args.device), indent=2)
    if args.output:
        target = Path(args.output)
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(report+'\n', encoding='utf-8')
    print(report)
