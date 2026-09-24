"""NumPy-only exchange helpers for small, serial reference meshes."""
import json
import numpy as np

TERMS = ("passive", "active", "pressure", "spring", "total")
EDGES = ((0, 1), (0, 2), (0, 3), (1, 2), (1, 3), (2, 3))


def match_points(query, candidates, tolerance=1e-11):
    """Unique coordinate match for this tiny fixture, never a production search.

    Reject missing/ambiguous matches instead of silently choosing nearest DOFs.
    """
    query, candidates = np.asarray(query), np.asarray(candidates)
    if (query.ndim != 2 or candidates.ndim != 2 or query.shape[1:] != (3,) or
            candidates.shape[1:] != (3,) or not len(query) or not len(candidates) or
            not np.isfinite(query).all() or not np.isfinite(candidates).all()):
        raise ValueError("expected finite nonempty 3D coordinate arrays")
    scale = max(1., float(np.ptp(candidates, axis=0).max()))
    near = np.linalg.norm(query[:, None]-candidates[None, :], axis=-1) <= tolerance*scale
    if not (near.sum(1) == 1).all():
        raise ValueError("missing or ambiguous coordinate match")
    indices = near.argmax(1)
    if len(np.unique(indices)) != len(indices):
        raise ValueError("coordinate mapping is not one-to-one")
    return indices


def load_reference(path):
    """Load numeric NPZ only; require explicit schema and reference provenance."""
    with np.load(path, allow_pickle=False) as archive:
        data = {key: archive[key] for key in archive.files}
    meta = json.loads(str(data.pop("metadata")))
    if meta.get("schema") != 1 or meta.get("producer") != "dolfinx-ufl":
        raise ValueError("unsupported reference schema or producer")
    N = len(data["X"])
    K = len(meta["cases"])
    shapes = {"X": (N, 3), "x": (K, N, 3), "direction": (N, 3),
              "fiber": (N, 3), "sheet": (N, 3), "tension": (N,),
              "pressure": (N,), "beta": (N,)}
    for term in TERMS:
        shapes["force_"+term] = shapes["tangent_"+term] = (K, N, 3)
    for name, shape in shapes.items():
        a = data[name]
        if a.shape != shape or a.dtype != np.float64 or not np.isfinite(a).all():
            raise ValueError(f"invalid reference array: {name}")
    for name, width in (("cells", 10), ("tagged_vertices", 3)):
        a = data[name]
        if (a.ndim != 2 or a.shape[1] != width or not len(a) or
                not np.issubdtype(a.dtype, np.integer) or (a < 0).any() or (a >= N).any()):
            raise ValueError(f"invalid connectivity: {name}")
    tags = data["facet_tags"]
    if tags.shape != (len(data["tagged_vertices"]),) or set(tags.tolist()) != {1, 2}:
        raise ValueError("expected nonempty pressure=1 and spring=2 facet tags")
    if K == 0 or len(set(meta["cases"])) != K:
        raise ValueError("expected distinct nonempty cases")
    return data, meta
