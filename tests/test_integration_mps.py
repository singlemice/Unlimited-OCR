"""End-to-end integration test: real model inference on Apple MPS (no CUDA, no triton).

This is the proof that baidu/Unlimited-OCR-2 runs locally on Apple Silicon through
the native Transformers path. The model ships triton kernels that are CUDA-only and
have no macOS wheel; crucially, every triton call site is gated behind ``is_cuda``
and falls back to a pure-Torch (eager) path on MPS. The only required source change
was removing a hardcoded ``.cuda()`` in the visual-feature scatter.

The test is skipped unless:
  * MPS is available on the host,
  * the vendored ``local_model/`` directory exists,
  * ``triton`` is NOT importable (the real Mac-no-triton scenario — if a CUDA box
    has triton installed this test is irrelevant and skipped).
"""
import os

# The model uses a few ops without an MPS kernel; PyTorch transparently falls these
# back to CPU. Enable that so the run completes instead of raising.
os.environ.setdefault("PYTORCH_ENABLE_MPS_FALLBACK", "1")

import sys

import pytest

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(REPO_ROOT, "utils"))

import torch  # noqa: E402

mps_available = (
    hasattr(torch.backends, "mps") and torch.backends.mps.is_available()
)
local_model_dir = os.path.join(REPO_ROOT, "local_model")
has_local_model = os.path.isdir(local_model_dir)
try:
    import triton  # noqa: F401  (only importable on CUDA/Linux)

    triton_importable = True
except Exception:
    triton_importable = False

skip_reason = None
if not mps_available:
    skip_reason = "MPS not available on this host"
elif not has_local_model:
    skip_reason = "vendored local_model/ not found (download with `huggingface-cli download baidu/Unlimited-OCR-2 --local-dir local_model`)"
elif triton_importable:
    skip_reason = "triton is importable here (CUDA/Linux scenario; MPS-no-triton path not exercised)"


@pytest.fixture(scope="module")
def loaded():
    if skip_reason:
        pytest.skip(skip_reason)
    import infer_transformers as it

    model, tokenizer, device = it.load_model(device="mps")
    assert device == "mps", f"expected mps, got {device}"
    return it, model, tokenizer, device


def test_load_model_on_mps(loaded):
    it, model, tokenizer, device = loaded
    # The model must be physically placed on the MPS device.
    param = next(model.parameters())
    assert param.device.type == "mps"


def test_infer_single_returns_text_meta_tuple(loaded):
    it, model, tokenizer, device = loaded
    fixture = os.path.join(REPO_ROOT, "tests", "fixtures", "sample_page.png")
    result = it.infer_single(
        fixture, model, tokenizer, device, output_path="output", save_results=False
    )
    # Stable 2-tuple contract: (markdown_text, meta=None).
    assert isinstance(result, tuple) and len(result) == 2, (
        f"infer_single must return (text, meta); got {type(result)!r}"
    )
    text, meta = result
    assert isinstance(text, str) and text.strip(), "OCR output must be non-empty"
    assert meta is None
    # The fixture is a synthetic page whose title is known; the OCR must recover it.
    assert "Unlimited-OCR-2 MPS Test Page" in text, (
        "expected the synthetic title to be recognised by MPS inference"
    )
