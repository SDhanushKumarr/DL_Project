# YOLO-World-Lite: Built From Scratch

A lightweight reimplementation of [YOLO-World](https://arxiv.org/abs/2401.17270)'s core open-vocabulary object detection design, trained on a real COCO subset from scratch.

---

## Overview

This project (Phase 2) replicates YOLO-World's key architectural innovations using a lightweight ResNet18 backbone and frozen CLIP text encoder. It demonstrates that **vision-language fusion via contrastive loss** — not just the architecture — is what enables open-vocabulary detection behavior.

**Key contribution:** Even a lightweight model trained on 5,000 images and 10 classes can exhibit open-vocabulary detection, generalizing to novel class names never seen during training (e.g., `"vehicle"`, `"human"`, `"animal"`), purely via CLIP's shared text-vision embedding space.

---

## Features

- ✅ YOLO-World-Lite architecture built from scratch (ResNet18 backbone + contrastive head)
- ✅ Frozen CLIP (ViT-B/32) as the text encoder — same as the original paper
- ✅ Region-text contrastive loss for classification — same as the original paper
- ✅ Trains on COCO val2017 (5,000 images, 10 classes) in ~2 hours on a free GPU
- ✅ Open-vocabulary inference — query any class name at test time, no retraining needed
- ✅ Early stopping, data augmentation, and separate learning rates for stable training

---

## Architecture

```
Input Image (416×416)
       │
  ResNet18 Backbone (pretrained, layer3+4 fine-tuned)
       │  (B, 512, 13, 13)
  Neck (Conv + BN + Dropout)
       │  (B, 256, 13, 13)
  ┌────┴────────────┬──────────────────┐
  │                 │                  │
Box Head        Obj Head          Region Feat Head
(tx,ty,tw,th)  (objectness)      (B, A, G, G, 512)
                                       │
                               L2 Normalize
                                       │
                               Cosine Similarity
                                       │
                           Frozen CLIP Text Embeddings
                           (encode any class at runtime)
                                       │
                               Class Scores (open-vocab)
```

---

## Dataset

- **Source:** COCO val2017 (~5,000 images)
- **Classes (10):** `person`, `car`, `dog`, `cat`, `bicycle`, `motorcycle`, `bus`, `truck`, `chair`, `bottle`
- **Split:** 80% train / 20% val
- Images are downloaded automatically during setup

---

## Requirements

```bash
pip install torch torchvision matplotlib pillow numpy
pip install git+https://github.com/openai/CLIP.git
```

A CUDA-capable GPU is strongly recommended. The notebook runs on Google Colab (free tier) in approximately 2 hours.

---

## Usage

Run the notebook cells in order:

| Step | Description |
|------|-------------|
| 0 | Install CLIP, restart kernel |
| 1 | Imports and device setup |
| 2 | Download COCO val2017 images and annotations |
| 3 | Build Dataset and DataLoader with augmentation |
| 4 | Pre-compute frozen CLIP text embeddings |
| 5 | Define `YOLOWorldLite` model |
| 6 | Define 3-part loss (box + objectness + contrastive) |
| 7 | Train with early stopping and cosine LR schedule |
| 8 | Plot training/validation loss curves |
| 9 | Qualitative detection visualizations |
| 10 | Quantitative evaluation (Recall@1 per class) |
| 11 | Open-vocabulary inference with custom class queries |
| 12 | Save all results to CSV and image files |

---

## Training Details

| Hyperparameter | Value |
|---|---|
| Input resolution | 416 × 416 |
| Grid size | 13 × 13 |
| Anchors | 3 per cell |
| Backbone | ResNet18 (ImageNet pretrained) |
| Backbone LR | 1e-4 |
| Head LR | 5e-4 |
| Optimizer | AdamW |
| Scheduler | Cosine Annealing |
| Max epochs | 20 |
| Early stopping patience | 6 |
| Batch size | 16 |
| Loss weights | box=5.0, obj=1.0, noobj=0.5, cls=1.0 |

**Key fixes applied over a naive baseline:**
1. **Data augmentation** (random flip, color jitter) on train split only — fixes val loss divergence
2. **Dropout2d** in neck (0.2) and region feature head (0.3) — prevents overfitting
3. **Dynamic class count** in `forward()` — enables open-vocabulary inference at any query size
4. **Empty mask guards** in loss — prevents NaN from propagating through training
5. **Label smoothing (0.1)** in cross-entropy — reduces overconfident class predictions
6. **`ACTUAL_EPOCHS` tracking** — correct plot/CSV export when early stopping fires

---

## Output Files

| File | Description |
|---|---|
| `best_model.pth` | Best model weights (lowest val loss) |
| `training_curves.png` | Train/val loss curves (total, obj, box, contrastive) |
| `detection_1-4.png` | Qualitative detection results on val images |
| `ovd_test_standard.png` | Open-vocab inference with standard class names |
| `ovd_test_novel.png` | Open-vocab inference with novel class names |
| `recall_results.csv` | Per-class Recall@1 on 100 val images |
| `training_history.csv` | Full epoch-by-epoch loss history |

---

## Open-Vocabulary Inference

A key feature of this model is that you can query **any class** at inference time — even classes never seen during training — by encoding them with CLIP on the fly:

```python
infer_with_custom_query(img_path, ['vehicle', 'human', 'animal'])
```

This works because the region feature head is trained to align with CLIP's text embedding space, enabling zero-shot generalization to novel vocabulary.

---

## Results

Evaluation metric: **Recall@1** — fraction of val images where at least one correct class is detected (evaluated on 100 val images, confidence threshold = 0.25).

Results are saved to `recall_results.csv` after running Step 10.

---

## Acknowledgements

- [YOLO-World (Cheng et al., 2024)](https://arxiv.org/abs/2401.17270) — original paper
- [OpenAI CLIP](https://github.com/openai/CLIP) — text encoder
- [COCO Dataset](https://cocodataset.org) — training data
