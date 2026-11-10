# Test Report — Cross-platform (CUDA / Apple MPS / CPU) Support for Unlimited-OCR 2

- **Date:** 2026-11-10
- **Author:** automated (Team Lead)
- **Environment:**
  - Host: Apple Mac mini, **Apple M4** (arm64), macOS 25.5 (Darwin 25.5.0)
  - Python: 3.10.17 (uv-managed `.venv`)
  - torch: 2.14.1 · transformers: 4.52.1 · pytest: 9.2.0
  - CUDA: **not available** on this host · MPS: **available** · CPU: available
- **Scope:** Verify the modifications that add Apple-Silicon (MPS) support and keep
  CUDA/CPU working, plus the pre-existing post-processing pipeline — **including a
  real end-to-end model run on Apple M4 MPS.**

## What changed (summary)
1. **`uv` virtual environment** created with `uv venv --python 3.10` (`pyproject.toml` added).
2. **`utils/device.py`** (new): `detect_device()` auto-selects `cuda > mps > cpu`;
   `resolve_torch_dtype()` returns `bfloat16` on accelerators and `float32` on CPU.
3. **`utils/infer_transformers.py`** (new): device-agnostic native-Transformers loader
   that replaces the upstream hardcoded `.cuda()`. Exposes `infer_single` / `infer_multi`
   (stable `(text, None)` 2-tuple contract) and a CLI with `--device {cuda,mps,cpu}`.
4. **`infer_mps.py`** (new): local inference entry point reusing `infer_transformers`,
   defaulting to the vendored `local_model/` (offline-friendly).
5. **`local_model/`** (vendored, 6.4 GB): the `baidu/Unlimited-OCR-2` model snapshot
   (config + code + safetensors). `modeling_unlimitedocr.py:785,789` had its hardcoded
   `.cuda()` removed (the only required model-source change for Mac).
6. **`README.md`** updated: device-agnostic setup, MPS instructions, offline vendored
   model usage, `--device cpu` fallback.
7. **`tests/`** (new): 22 tests — device selection, post-processing, MPS-support
   contract, and a real MPS integration test.

> The **SGLang serving path** (`infer.py` + the custom `wheel/*.whl`) is CUDA-only and
> intentionally left untouched — it talks to a remote/served model and cannot run on
> MPS. MPS support is delivered through the native **Transformers** path.

## Why no triton rewrite was needed (the key finding)
`baidu/Unlimited-OCR-2`'s block-sparse attention ships **CUDA-only triton kernels** with
no macOS wheel. However, **every triton call site is gated behind `is_cuda`** and falls
back to a pure-Torch *eager* dense block-mask path on MPS/CPU — functionally equivalent
for inference. The module-level `import triton` is wrapped in `try/except`, so it does
not raise when triton is absent. Consequently **MPS inference works without rewriting any
kernel**; the only model-source edit required was removing the two unconditional
`.cuda()` calls in the visual-feature scatter (`modeling_unlimitedocr.py:785,789`).

## Test results

```
platform darwin -- Python 3.10.17, pytest-9.2.0
rootdir: /Volumes/Project/Unlimited-OCR
configfile: pyproject.toml
collected 22 items

tests/test_device.py ......                                              [ 27%]
tests/test_integration_mps.py ..                                         [ 36%]
tests/test_mps_support.py ....                                           [ 54%]
tests/test_postprocess.py ..........                                     [100%]

============================= 22 passed, 1 warning in 37.31s ==============================
```

### By file

| File | Tests | Result |
|------|-------|--------|
| `tests/test_device.py` | 6 | ✅ pass |
| `tests/test_integration_mps.py` | 2 | ✅ pass (real 6.4 GB model loaded on MPS) |
| `tests/test_mps_support.py` | 4 | ✅ pass |
| `tests/test_postprocess.py` | 10 | ✅ pass |

### Notable assertions
- **Device selection** — explicit `cuda`/`mps`/`cpu` honored; invalid preference falls
  back to auto; CPU→`float32`, accelerator→`bfloat16`; `--device` arg wiring.
- **MPS on Apple Silicon** — auto-selection picks `mps` when CUDA is absent
  (skipped automatically on non-Apple hosts).
- **No forced CUDA** — `infer.py` / `utils/infer_transformers.py` contain no
  `.cuda()` *call*; the only CUDA reference is the detection branch in `utils/device.py`.
- **Helper wiring** — `infer_transformers` imports, exposes `load_model` /
  `infer_single` / `infer_multi`, and delegates device choice to `detect_device`.
- **Post-processing** — title/level → heading, page ordering, image Markdown with
  coordinate clamping to 0–1000, cross-page table merge with repeated-header drop,
  stray `<|...|>` token stripping, code-fence stripping, single-root heading collapse,
  and union-find grouping of `cross_`/`append_` chains.
- **Real MPS integration** — loads the vendored model on `device='mps'`, asserts
  `param.device.type == 'mps'`, runs `infer_single`, and asserts the recovered Markdown
  contains the known synthetic-page title `Unlimited-OCR-2 MPS Test Page`.

## Manual / integration verification
- ✅ **Device resolution smoke**: `detect_device(None)` → `mps`, `resolve_torch_dtype`
  → `bfloat16` on this M4.
- ✅ **CLI**: `python infer_mps.py --help` / `python utils/infer_transformers.py --help`
  render the new options (`--device`, `--model`, `--image`, `--images`, `--out`,
  `--no-save`).
- ✅ **MPS backend executes real compute**: a 4×4 matmul on `device='mps'` returned a
  finite result.
- ✅ **End-to-end inference on Apple M4 MPS — RUN AND PASSED.** The vendored
  `baidu/Unlimited-OCR-2` loaded on MPS (≈14 s), the DASH block-sparse attention took
  its **differentiable eager fallback** path (`[DASH] differentiable DASH attention
  active`), and `infer_single` produced **correct OCR** on a synthetic page — title,
  body text, an invoice total (`1,234.56 CNY`) and a line item (`1 x 36.50`) were all
  read back accurately. Load + single-page inference ≈ 33 s. Output captured in
  `tests/fixtures/mps_infer_output.txt`.
- ⚠️ **`mpsgraph` cache-dir permission warning** appears on this sandboxed host (write
  access to the MPS compiled-graph cache is denied). It is **harmless** — PyTorch falls
  back to compiling at runtime and continues. On a normal machine this warning is absent.
- ⚠️ **CUDA path not executed here** (no NVIDIA GPU on this host). The same
  device-selection logic and the upstream `.cuda()`-free placement cover it; it should
  still be re-run on a CUDA machine before release.

## How to run
```shell
source .venv/bin/activate
# (deps already installed via uv into .venv)
pytest tests/ -v

# One real MPS inference (vendored offline model):
PYTORCH_ENABLE_MPS_FALLBACK=1 python infer_mps.py --image tests/fixtures/sample_page.png
# Or point at the Hugging Face id with --model baidu/Unlimited-OCR-2
```

> On this sandbox, also export `HF_HOME=/Volumes/Project/Unlimited-OCR/.hf-cache` and
> `UV_CACHE_DIR=/Volumes/Project/Unlimited-OCR/.uv-cache` (the default `~/.cache` paths
> are not writable here).

## Verdict
✅ **All 22 automated tests pass, including a real end-to-end model inference on Apple M4
MPS.** The repository now supports Apple M4 (MPS) in addition to CUDA and CPU. No triton
kernel rewrite was required — the model's built-in `is_cuda`-gated eager fallback plus a
single `.cuda()` removal suffice. The only remaining caveat is environmental (a CUDA
re-run belongs on an NVIDIA machine), not a code defect.
