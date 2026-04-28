# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

Unofficial PyTorch implementation of [WinCLIP (CVPR 2023)](https://arxiv.org/abs/2303.14814) — zero-/few-shot anomaly classification and segmentation using CLIP-based models on industrial inspection datasets (MVTec-AD and VisA).

## Setup

```bash
sh install.sh   # installs PyTorch 1.10 + CUDA 11.3, transformers, open_clip_torch, etc.
```

## Running Evaluations

**Single class, zero-shot:**
```bash
python eval_WinCLIP.py --dataset mvtec --class-name carpet
python eval_WinCLIP.py --dataset visa  --class-name candle --k-shot 5
```

**Key flags:**
- `--dataset` — `mvtec` or `visa`
- `--class-name` — category name (see `datasets/mvtec.py` or `datasets/visa.py` for lists)
- `--k-shot` — 0 (zero-shot), 1, 5, or 10
- `--backbone` / `--pretrained_dataset` — CLIP backbone (default: `ViT-B-16-plus-240` / `laion400m_e32`)
- `--vis` — save anomaly map visualizations
- `--cal-pro` — compute per-region overlap (PRO) metric (slow)
- `--gpu-id` / `--use-cpu` — device selection

**Batch evaluation (all classes, multiprocessing):**
```bash
python run_winclip.py
```
Edit `run_winclip.py` to configure dataset, k-shot value, and number of parallel processes.

## Architecture

### Data flow

```
eval_WinCLIP.py: test()
  ├── model.build_text_feature_gallery(category)
  │     └── encode 154 prompts → normal_text_features, abnormal_text_features
  ├── [k-shot > 0] model.build_image_feature_gallery(train_data)
  │     └── pool multi-scale features from normal training images → visual_gallery
  └── for each test image:
        model.forward(image)
          ├── encode_image() → multi-scale sliding-window features
          ├── calculate_textual_anomaly_score()  → text-cosine similarity map
          ├── calculate_visual_anomaly_score()   → gallery nearest-neighbor map
          └── fuse (harmonic mean) → 400×400 anomaly map
```

### Key modules

| Module | Role |
|--------|------|
| `WinCLIP/model.py` | `WinClipAD` — all inference logic, gallery management, score fusion |
| `WinCLIP/ad_prompts.py` | 22 template phrases × state-level normal/abnormal descriptors (~154 prompts total) |
| `WinCLIP/CLIPAD/` | Modified OpenCLIP backbone — multi-scale ViT feature extraction, model configs (60+ JSON files), tokenizer, transforms |
| `datasets/` | `CLIPDataset` (PyTorch Dataset), MVTec-AD and VisA loaders, k-shot seed splits |
| `utils/metrics.py` | AUROC, F1, PRO calculation |
| `utils/visualization.py` | cv2-based heatmap overlay |
| `utils/csv_utils.py` | Per-class result logging to CSV |

### Multi-scale feature extraction

`WinCLIP/model.py` uses a sliding-window approach: the ViT image encoder is called on crops at multiple resolutions to produce a stack of spatial feature maps. These are masked (`self.masks`) to handle boundary crops, then used for both text-cosine and gallery-nearest-neighbor scoring.

### Score fusion

`fuse_scores()` combines the textual and visual anomaly scores via harmonic mean. When `k-shot=0`, only the textual path is active.

## Dataset Setup

- **MVTec-AD**: standard layout, set path with `--data-path`
- **VisA**: run `python datasets/prepare_visa_public.py` to restructure the public download into the expected layout
- k-shot splits are seeded — seeds stored under `datasets/seeds_mvtec/`
