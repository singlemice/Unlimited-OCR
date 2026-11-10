#!/usr/bin/env python3
"""Native (Transformers) inference for Unlimited-OCR 2 on CUDA / Apple MPS / CPU.

The upstream example places the model with a forced CUDA call, which only runs on
NVIDIA GPUs. This module is the device-agnostic equivalent: it loads the model
onto the best available accelerator (CUDA > MPS > CPU) and exposes the same
``infer`` (single page) and ``infer_multi`` (multi-page) entry points the model
provides, plus a small CLI.

Example
-------
    python utils/infer_transformers.py --image your_doc/page_1.jpg --out output
    python utils/infer_transformers.py --images page_1.jpg page_2.jpg --out output

Notes
-----
* The SGLang serving path (``infer.py`` + the custom wheel) remains CUDA-only; it
  talks to a server and is unaffected by this module.
* Apple MPS (M1/M2/M3/M4) runs in ``torch.bfloat16`` by default; some custom
  modelling ops may still fall back to CPU for a given kernel, which PyTorch
  handles transparently via ``torch.device`` placement.
"""
from __future__ import annotations

import argparse
import os
import sys

# Allow running as a sibling import when executed from the repo root.
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from device import add_device_arguments, detect_device, resolve_torch_dtype  # noqa: E402

# Default to the vendored, triton-free copy under local_model/ so MPS/CPU work
# fully offline. Override with --model baidu/Unlimited-OCR-2 when CUDA+network
# are available.
MODEL_NAME = "local_model"


def load_model(model_name: str = MODEL_NAME, device: str | None = None, dtype=None):
    """Load the Unlimited-OCR 2 model onto the requested/supported device.

    Returns ``(model, tokenizer, device)``. Unlike the upstream example this does
    NOT call ``.cuda()``; it uses the detected device (cuda / mps / cpu).
    """
    import torch
    from transformers import AutoModel, AutoTokenizer

    device = detect_device(device)
    if dtype is None:
        dtype = resolve_torch_dtype(device)

    tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
    model = AutoModel.from_pretrained(
        model_name,
        trust_remote_code=True,
        use_safetensors=True,
        torch_dtype=dtype,
    )
    model = model.eval().to(device)
    return model, tokenizer, device


def infer_single(image_file: str, model=None, tokenizer=None, device: str | None = None,
                 output_path: str = "output", save_results: bool = True):
    """Parse a single page image (dynamic tiling / gundam mode by default).

    Returns ``(text, meta)`` where ``text`` is the OCR markdown string and
    ``meta`` is ``None`` (reserved for future structured output). This stable
    2-tuple keeps callers that unpack ``text, _ = infer_single(...)`` working
    regardless of whether the underlying ``model.infer`` returns a bare string.
    """
    if model is None or tokenizer is None:
        model, tokenizer, device = load_model(device=device)
    result = model.infer(
        tokenizer,
        image_file=image_file,
        output_path=output_path,
        save_results=save_results,
    )
    return _as_text_meta(result)


def infer_multi(image_files, model=None, tokenizer=None, device: str | None = None,
                output_path: str = "output", save_results: bool = True):
    """Parse all pages of a document in one request (cross-page context).

    Returns ``(text, meta)`` — see :func:`infer_single`.
    """
    if model is None or tokenizer is None:
        model, tokenizer, device = load_model(device=device)
    result = model.infer_multi(
        tokenizer,
        image_files=list(image_files),
        output_path=output_path,
        save_results=save_results,
    )
    return _as_text_meta(result)


def _as_text_meta(result):
    """Normalise the model's ``infer``/``infer_multi`` return into ``(text, None)``.

    Upstream ``model.infer`` returns a markdown string; guard against it ever
    returning a tuple so the public 2-tuple contract stays fixed.
    """
    if isinstance(result, tuple):
        text = result[0]
    else:
        text = result
    return text, None


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    add_device_arguments(ap)
    ap.add_argument("--model", default=MODEL_NAME, help="HuggingFace model id or local path")
    g = ap.add_mutually_exclusive_group(required=True)
    g.add_argument("--image", help="a single page image")
    g.add_argument("--images", nargs="+", help="multiple page images (one document)")
    ap.add_argument("--out", default="output", help="output directory")
    ap.add_argument("--no-save", action="store_true", help="do not write results to disk")
    args = ap.parse_args()

    model, tokenizer, device = load_model(model_name=args.model, device=args.device)
    print(f"[device] using {device} (dtype={resolve_torch_dtype(device)})", flush=True)

    save = not args.no_save
    if args.image:
        text, _ = infer_single(args.image, model, tokenizer, device, args.out, save)
        print(text)
    else:
        text, _ = infer_multi(args.images, model, tokenizer, device, args.out, save)
        print(text)


if __name__ == "__main__":
    main()
