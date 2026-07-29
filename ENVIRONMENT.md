# Environment Notes — AAGNet Local Adaptation

Date: 2026-07-29
Machine: Desktop workstation (CPU: Intel(R) Core(TM) Ultra 7 270K Plus, GPU: NVIDIA GeForce RTX 5070 12 GB, OS: Windows 10/11 64-bit), conda env: `aagnet`

## 1. Base environment

- Base source: cloned from local env `uvnet` (`conda create -n aagnet --clone uvnet`), which was previously validated for the UV-Net / BRepNet baselines on the same machine.
- Key inherited versions:

  | Package | Version |
  |---|---|
  | python | 3.10.20 |
  | torch | 2.11.0+cu128 |
  | torchvision | 0.26.0+cu128 (added for timm, see §3) |
  | dgl | 2.2.1+cu121 |
  | numpy | 1.23.5 |
  | pytorch-lightning | 1.9.5 (installed but unused by AAGNet) |
  | occwl | 3.0.0 |
  | occt / pythonocc-core | 7.7.2 |

- Additional installs on top of the clone: `wandb 0.28.1`, `timm 0.9.2 (--no-deps)`, `huggingface-hub`, `safetensors`, `torch-ema 0.3 (--no-deps)`, `torchmetrics 1.9.0`, `torchvision 0.26.0+cu128`, `ijson 3.5.1`.
- Full fingerprint: see `environment-adapted.yml` in this repo (exported via `conda env export -n aagnet`, UTF-8).

## 2. Differences vs. official environment.yml

The official `environment.yml` is a full Windows machine snapshot (300+ packages, incl. author's dev tools). Only the training-path subset was reproduced. Version mapping:

| Dependency | Official pin | This env | Note |
|---|---|---|---|
| python | 3.10.8 | 3.10.20 | same minor |
| pytorch | 2.0.1+cu118 | 2.11.0+cu128 | major upgrade, verified |
| dgl | 1.1.0.cu118 | 2.2.1+cu121 | all used APIs stable across 1.1→2.2 (verified by grep audit + smoke run) |
| occwl | 2.0.2 | 3.0.0 | only used for data generation / STEP viz, not hit in smoke |
| occt / pythonocc-core | 7.5.1 | 7.7.2 | same as above |
| lightning / pytorch-lightning | 2.0.3 | 1.9.5 | AAGNet seg_trainer does NOT import PL (verified); irrelevant |
| torchmetrics | 0.11.4 | 1.9.0 | only `MulticlassAccuracy` / `MulticlassJaccardIndex` used; API-compatible |
| timm | 0.9.2 | 0.9.2 | same; only `DropPath` used |
| torch-ema | 0.3 | 0.3 | same |
| wandb | 0.15.5 | 0.28.1 | basic `init/config` usage only, compatible |
| numpy | 1.24.3 | 1.23.5 | compatible |
| torchvision | 0.15.2 | 0.26.0+cu128 | required at timm import time (`timm.layers` → `torchvision.ops.misc`) |
| numba, h5py, cupy, opencv, selectivesearch, pyvista, pyqt/pyside2/wxpython | pinned/listed | NOT installed | data-generation & visualization only; out of smoke scope |
| ijson | — | 3.5.1 | added: streaming JSON extraction tool (§6, commit 98103eb) |

## 3. Key dependencies

- **dgl 2.2.1+cu121**: inherited from `uvnet`; originally installed via `pip install dgl -f https://data.dgl.ai/wheels/cu121/repo.html` (PyPI default serves CPU-only — do NOT use it). Requires the graphbolt stub (§4).
- **torchvision**: MUST come from the PyTorch cu128 index to match torch 2.11: `pip install torchvision --index-url https://download.pytorch.org/whl/cu128`. pip's only-if-needed strategy picks the cu128 build pairing torch 2.11.
- **timm 0.9.2**: install with `--no-deps`; its declared `torchvision` dep is real at import time (see above) but must be satisfied manually with the cu128 build.
- **torch-ema 0.3**: install with `--no-deps` (pure-python, torch only).
- **torchmetrics**: plain `pip install torchmetrics`; NEVER use `-U` (see incident below).
- **ijson 3.5.1**: used by `tools/extract_smoke.py`; the C backend (`yajl2_c`) is present and much faster; pure-python backend returns `Decimal` (handled via `json.dump(..., default=float)`).
- ⚠️ **Incident recorded (2026-07-29)**: running `pip install "timm==0.9.2"` with deps / `pip install -U torchmetrics` pulls the latest `torchvision`/`torch` from default PyPI and silently REPLACES torch 2.11.0+cu128 with a CPU build (observed: torch 2.13.0+cpu, dgl then fails to load `dgl.dll`). Recovery: delete env, re-clone from `uvnet`, reinstall with `--no-deps` where applicable. Operational rules: (a) check the prompt shows the target env before pip; (b) after every pip install, inspect the "Installing collected packages" list — abort immediately if `torch`/`torchvision`/`numpy` appears.

## 4. Environment-level patches

- **dgl graphbolt stub** (inherited from `uvnet` via clone): `dgl/graphbolt/__init__.py` in site-packages is replaced with a `__getattr__` stub returning `None`, because graphbolt only supports torch 2.1–2.3 while this env runs torch 2.11. Reason: `import dgl` otherwise fails. **Must be re-applied after any dgl reinstall/upgrade.**

## 5. Verification log

| Level | Check | Result |
|---|---|---|
| import | `torch.__version__, cuda.is_available()` | `2.11.0+cu128, True` |
| import | `dgl.__version__` (cold start, no torch first) | `2.2.1+cu121` |
| import | `from timm.models.layers import DropPath` | ok (after torchvision cu128 install) |
| import | `import wandb, torch_ema, torchmetrics` | ok |
| preprocess | `tools/extract_smoke.py` on official MFCAD++ gAAG (20.85 GB graphs.json) | extracted 500/500 graphs, 0 missing labels |
| smoke | `python -m engine.seg_trainer`, MFCAD2_smoke subset (300/100/100), 10 epochs, batch 64, RTX 5070 | train 299 (1 zero-edge graph dropped by official filter), val/test 100 each; loss 2.43→1.17 monotone; test_seg_acc **30.8%**, test_seg_iou **18.9%** |
| reference frame | same-machine UV-Net smoke | IoU 61.17% (different subset size/epochs — not directly comparable); 25-class random ≈ 4% acc, so 30.8%/18.9% with monotone val improvement across all 10 epochs = healthy pipeline signal at only 40 optimizer steps |

Smoke verdict: PASS (entry point runs, learning curve monotone, test metrics in sane range, checkpoints/logging/wandb all functional).

Known non-issues: (a) `UserWarning: Converting a tensor with requires_grad=True to a scalar` at logging line — cosmetic; (b) `Done loading 299 files` — one graph filtered by official zero-edge guard; (c) test metrics are written to `output/&lt;timestamp&gt;/log.txt`, not printed to console.

Outstanding (for formal experiments, NOT smoke): full `graphs.json` (20.85 GB) cannot be `json.load`-ed within 32 GB RAM (needs ~60–100 GB); a streaming/chunked data-loading strategy must be decided together with the preprocessing strategy.

## 6. Code-level adaptations

| # | File | Change | Commit |
|---|---|---|---|
| 1 | `engine/seg_trainer.py` | smoke config: dataset path → `D:/ShortEssay/Datasets/MFCAD2_smoke`, epochs 100→10, batch_size 256→64 | `fb16fc2` |
| 2 | `tools/extract_smoke.py` | NEW: ijson-based streaming extractor building the MFCAD2_smoke subset (300/100/100) from official graphs.json | `98103eb` |
| 3 | `.gitignore` | NEW: ignore `__pycache__/`, `*.pyc`, `output/`, `wandb/` | `c99f0c9` |
| 4 | `utils/data_utils.py` | CONDITIONAL, NOT APPLIED: `torch.load(pkl_path)` would need `weights_only=False` under torch ≥2.6 — official dataset ships no `.pkl`, so the `graphs.json` branch is taken and no change was made | — |

No changes were required for dgl 2.2.1 (all call sites use stable APIs: `dgl.batch`, `dgl.nn.*`, `dgl.function`, `edge_softmax`) or torchmetrics 1.9.0 (`MulticlassAccuracy`/`MulticlassJaccardIndex` signatures unchanged).