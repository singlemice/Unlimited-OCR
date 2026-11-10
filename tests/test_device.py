"""Tests for cross-platform device selection (utils/device.py)."""
import importlib.util
import os
import sys

import pytest

# Make utils importable as a package without installing it.
REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(REPO_ROOT, "utils"))

import device  # noqa: E402


def test_detect_explicit_values():
    assert device.detect_device("cuda") == "cuda"
    assert device.detect_device("mps") == "mps"
    assert device.detect_device("cpu") == "cpu"


def test_detect_invalid_preferred_falls_through():
    # An unrecognised preference must not crash; it triggers auto-selection.
    assert device.detect_device("nonsense") in ("cuda", "mps", "cpu")


def test_dtype_cpu_is_float32():
    import torch

    assert device.resolve_torch_dtype("cpu") is torch.float32


def test_dtype_accelerator_is_bfloat16():
    import torch

    # Both CUDA and MPS use the training dtype (bfloat16).
    assert device.resolve_torch_dtype("cuda") is torch.bfloat16
    assert device.resolve_torch_dtype("mps") is torch.bfloat16


def test_add_device_arguments():
    import argparse

    parser = argparse.ArgumentParser()
    device.add_device_arguments(parser)
    ns = parser.parse_args(["--device", "mps"])
    assert ns.device == "mps"


@pytest.mark.skipif(
    not (getattr(__import__("torch").backends, "mps", None) is not None
         and __import__("torch").backends.mps.is_available()),
    reason="MPS not available on this host",
)
def test_auto_selects_mps_on_apple_silicon():
    # On an Apple-Silicon machine without CUDA, auto-selection must pick MPS.
    assert device.detect_device(None) == "mps"
