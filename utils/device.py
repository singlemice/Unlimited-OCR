"""Cross-platform device selection for Unlimited-OCR 2 inference.

The upstream project hardcodes ``.cuda()`` in its Transformers example, which only
works on NVIDIA GPUs. This helper lets the same code run on:

* NVIDIA CUDA  (``cuda``)
* Apple Silicon / M4  (``mps``)
* any CPU fallback  (``cpu``)

Selection priority: explicit ``--device`` argument -> CUDA if available ->
MPS if available (Apple Silicon) -> CPU.
"""
from __future__ import annotations

import argparse


def detect_device(preferred: str | None = None) -> str:
    """Return the torch device string to use.

    Args:
        preferred: one of ``"cuda"``, ``"mps"``, ``"cpu"`` or ``None``.
            When ``None`` (or unrecognised) the best available device is chosen
            automatically: CUDA > MPS > CPU.

    Returns:
        A torch device string: ``"cuda"``, ``"mps"`` or ``"cpu"``.
    """
    if preferred in ("cuda", "mps", "cpu"):
        return preferred

    # CUDA
    try:
        import torch

        if torch.cuda.is_available():
            return "cuda"
    except Exception:
        pass

    # Apple Metal (M1/M2/M3/M4 ...)
    try:
        import torch

        if getattr(torch.backends, "mps", None) is not None and torch.backends.mps.is_available():
            # MPS has known issues with some ops; fall back is handled per-call.
            return "mps"
    except Exception:
        pass

    return "cpu"


def resolve_torch_dtype(device: str) -> "object":
    """Pick a sensible default dtype per device.

    * CUDA / MPS: ``torch.bfloat16`` (the dtype the model was trained with).
    * CPU: ``torch.float32`` (bf16 on CPU is slow / unsupported in places).
    """
    import torch

    if device == "cpu":
        return torch.float32
    return torch.bfloat16


def add_device_arguments(parser: argparse.ArgumentParser) -> None:
    """Add a ``--device`` argument to an argument parser."""
    parser.add_argument(
        "--device",
        default=None,
        choices=["cuda", "mps", "cpu"],
        help="Compute device. Default: auto (cuda > mps > cpu).",
    )


def warmup_mps() -> None:
    """Allocate a tiny tensor on MPS to avoid the first-call lazy-init stall."""
    import torch

    if torch.backends.mps.is_available():
        try:
            _ = torch.zeros(1, device="mps")
        except Exception:
            pass
