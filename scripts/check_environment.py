"""Run with --device cuda on the target machine; never silently falls back."""
import argparse
import json
import platform

import numpy as np
import torch


def check(device):
    if device.startswith("cuda") and not torch.cuda.is_available():
        raise RuntimeError("CUDA requested but unavailable; check PyTorch build and driver")
    report = {
        "python": platform.python_version(), "platform": platform.platform(),
        "torch": torch.__version__, "numpy": np.__version__,
        "cuda_runtime": torch.version.cuda, "device": device,
        "visible_gpus": [torch.cuda.get_device_name(i) for i in range(torch.cuda.device_count())],
    }
    F = torch.tensor([[1.1, .1, 0.], [0., .9, 0.], [0., 0., 1.05]],
                     dtype=torch.float64, device=device, requires_grad=True)
    logJ = torch.log(torch.linalg.det(F))
    W = (F.square().sum() - 3) - 2 * logJ + 2.5 * logJ.square()
    P, = torch.autograd.grad(W, F)
    invT = torch.linalg.inv(F).T
    expected = 2 * (F - invT) + 5 * logJ * invT
    torch.testing.assert_close(P, expected, atol=1e-11, rtol=1e-10)
    # Exercise sparse assembly/coalescing/matvec, needed by the later FEM solver.
    idx = torch.tensor([[0, 0, 1], [0, 0, 1]], device=device)
    vals = torch.tensor([1., 2., 4.], dtype=torch.float64, device=device)
    A = torch.sparse_coo_tensor(idx, vals, (2, 2), check_invariants=True).coalesce()
    y = torch.sparse.mm(A, torch.ones((2, 1), dtype=torch.float64, device=device))
    torch.testing.assert_close(y, F.new_tensor([[3.], [4.]]))
    report.update(float64_autograd="passed", sparse_coo_matvec="passed",
                  max_stress_error=float((P - expected).abs().max().detach().cpu()))
    return report


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--device", choices=["cpu", "cuda", "cuda:0", "cuda:1"], default="cpu")
    args = parser.parse_args()
    print(json.dumps(check(args.device), indent=2))
