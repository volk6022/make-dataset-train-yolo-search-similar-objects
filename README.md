# tools to train yolo and compare detected objects

Code for detecting the same objects across multiple independent images (originally developed for soil core images).
Includes notebooks and Python tools to prepare datasets for YOLO training (datasets labelled with CVAT).

`src/pipline_final.py` is the main entry point. Pipeline steps:

- load and format images
- detect objects with YOLO (Oriented Bounding Boxes)
- crop each detected object
- extract embedding from cropped object
- compare embeddings with a similarity threshold
- verify candidate pairs with feature matching

Or shorter: **detect → embed → search → verify**

The pipeline produces intermediate files at each step so individual stages can be re-run independently.

---

## Embedding models

Set `embedding_model_name` in `src/settings_and_utils.py`:

| Key | Model | Dim | Notes |
|-----|-------|-----|-------|
| `siglip` **(default)** | `google/siglip-so400m-patch14-384` | 1152D | 384px input captures fine layered detail; sigmoid-loss calibration gives well-spaced similarity scores |
| `dino` | `facebook/dinov2-large` | 1024D | Strong texture-aware features |
| `dino_giant` | `facebook/dinov2-giant` | 1536D | Larger DINOv2; better texture discrimination |
| `clip` | `openai/clip-vit-large-patch14` | 768D | Semantic visual similarity, L2-normalized |
| `radio` | `nvidia/RADIO` (via timm) | 768D | Multi-teacher distillation combining DINOv2 + CLIP + SAM |
| `vit_google` | `google/vit-base-patch16-224-in21k` | 768D | Baseline ViT |
| `sd-vae` | Stable Diffusion VAE | latent | Experimental; did not improve results in testing |

---

## Verification methods

Set `local_features_method` in `src/settings_and_utils.py`:

| Key | Backend | Notes |
|-----|---------|-------|
| `XFEAT` **(default)** | [XFeat](https://github.com/verlab/accelerated_features) via `torch.hub` | Neural keypoints, no install needed, CPU-friendly, handles repetitive textures better than SIFT |
| `LIGHTGLUE` | [LightGlue](https://github.com/cvg/LightGlue) + SuperPoint/DISK/ALIKED | State-of-the-art transformer matcher; extractor selectable via `lightglue_extractor` |
| `SIFT` | OpenCV SIFT + FLANN | Classical, reliable baseline |
| `ORB` | OpenCV ORB + BFMatcher | Faster than SIFT, binary descriptors |

A second image-features pass (EDGE/CORNER/BLOB/TEXTURE) runs in parallel — pairs passing either check are kept.

---

## All settings

All configuration lives in `src/settings_and_utils.py` (`Settings` class).

---

## Experimental / previous attempts

- **VAE fine-tuning** (`src/try_to_finetune_vae.py`): attempted to fine-tune a Stable Diffusion VAE as an embedding extractor. Did not improve results.
- **Siamese network** (`src/similarity_model_learning/`): small NN to map embeddings to a more Euclidean-comparable space. Limited by small dataset and ~95% class imbalance (non-matching pairs dominated).
- **Multi-metric comparison** (`src/image_comparator.py` + `src/test_comparing_methods.py`): tested 11 image similarity metrics (SSIM, NCC, histogram, edge, gradient, etc.).

---

## Debug output examples

`collage/`, `debug_image_features/`, `debug_local_features/`, `images_formated/`
