#!/usr/bin/env python3
"""Local (device-agnostic) inference entry point for Unlimited-OCR 2 — Apple MPS / CUDA / CPU.

This is the **MPS-friendly** counterpart of ``infer.py``.

``infer.py`` is the SGLang *serving client*: it imports the custom ``sglang`` wheel
(which is CUDA-only) and POSTs documents to a running server. It cannot run on Apple
Silicon because SGLang's serving path has no Metal backend.

This script instead runs the model **locally** through the native Transformers path,
auto-selecting the best accelerator via ``utils/device.py``:

    * NVIDIA CUDA  -> ``cuda``
    * Apple M4/M3/M2/M1 -> ``mps``
    * fallback     -> ``cpu``

It supports the same document shapes as ``infer.py``: a single page, a list of page
images (one document, parsed in one request for cross-page context), a ``--doc``
directory of ``page_<n>.(jpg|jpeg|png)``, and a ``--pdf`` (rasterised then parsed).

Examples
--------
    # single page
    python infer_mps.py --image your_doc/page_1.jpg --out output
    # a whole document (all pages in one request)
    python infer_mps.py --images page_1.jpg page_2.jpg --out output
    # a directory of page_N.jpg
    python infer_mps.py --doc your_doc --out output
    # a PDF (rasterised at 300 dpi)
    python infer_mps.py --pdf your_doc.pdf --out output
    # force a device
    python infer_mps.py --doc your_doc --device mps --out output
"""
import argparse
import os
import re
import sys
import tempfile

# Make the sibling utils importable when run from the repo root.
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "utils"))

from device import add_device_arguments, detect_device, resolve_torch_dtype  # noqa: E402
import infer_transformers as it  # noqa: E402

# Default to the vendored, triton-free copy under local_model/ (offline-friendly
# for Apple Silicon). Override with --model baidu/Unlimited-OCR-2 if you have a
# CUDA machine with network access.
MODEL_NAME = "local_model"
PAGE_RE = re.compile(r"page_\d+\.(jpg|jpeg|png)$", re.I)


def collect_doc_pages(doc_dir, max_pages=0):
    if not os.path.isdir(doc_dir):
        raise SystemExit(f"doc dir not found: {doc_dir}")
    files = [f for f in os.listdir(doc_dir) if PAGE_RE.match(f)]
    files.sort(key=lambda f: int(re.search(r"page_(\d+)", f, re.I).group(1)))
    if max_pages:
        files = files[:max_pages]
    if not files:
        raise SystemExit(f"no page_*.jpg/png found in: {doc_dir}")
    return [os.path.join(doc_dir, f) for f in files]


def pdf_to_images(pdf_path, dpi=300, out_dir=None):
    try:
        import fitz
    except ImportError:
        raise SystemExit("PyMuPDF is required for --pdf (uv pip install pymupdf==1.27.2.2)")
    doc = fitz.open(pdf_path)
    paths = []
    for i, page in enumerate(doc):
        tmp = (out_dir or tempfile.mkdtemp(prefix="pdf_ocr_"))
        os.makedirs(tmp, exist_ok=True)
        img = os.path.join(tmp, f"page_{i + 1:04d}.png")
        page.get_pixmap(matrix=fitz.Matrix(dpi / 72, dpi / 72)).save(img)
        paths.append(img)
    doc.close()
    return paths


def main():
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    add_device_arguments(ap)
    ap.add_argument("--model", default=MODEL_NAME, help="HuggingFace model id or local path")
    g = ap.add_mutually_exclusive_group(required=True)
    g.add_argument("--image", help="a single page image")
    g.add_argument("--images", nargs="+", help="multiple page images (one document)")
    g.add_argument("--doc", help="a directory of page_<n>.(jpg|jpeg|png)")
    g.add_argument("--pdf", help="a PDF file (rasterised then parsed)")
    ap.add_argument("--max-pages", type=int, default=0, help="only the first N pages, 0 = all")
    ap.add_argument("--out", default="output", help="output directory")
    ap.add_argument("--no-save", action="store_true", help="do not write results to disk")
    args = ap.parse_args()

    # Resolve the input image list.
    if args.image:
        images = [args.image]
    elif args.images:
        images = list(args.images)
    elif args.doc:
        images = collect_doc_pages(args.doc, args.max_pages)
    else:  # --pdf
        images = pdf_to_images(args.pdf, out_dir=args.out)

    model, tokenizer, device = it.load_model(model_name=args.model, device=args.device)
    print(f"[device] {device} (dtype={resolve_torch_dtype(device)}), "
          f"{len(images)} page(s)", flush=True)

    save = not args.no_save
    if save:
        os.makedirs(args.out, exist_ok=True)

    if len(images) == 1:
        text, _ = it.infer_single(images[0], model, tokenizer, device, args.out, save)
        print(text)
    else:
        text, _ = it.infer_multi(images, model, tokenizer, device, args.out, save)
        print(text)


if __name__ == "__main__":
    main()
