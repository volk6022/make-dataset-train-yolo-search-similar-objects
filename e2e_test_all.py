"""
E2E comparison test: runs every combination of
  embedding model × distance method × verification method
and saves results in separate folders for side-by-side comparison.

Output tree:
  e2e_results/
    summary.json
    {embedding}/
      predictions_with_embeddings.json
      {distance}/
        similarity.json
        local_XFEAT/    {similarity_verified.json, collages/, debug_matches/}
        local_LIGHTGLUE/
        local_SIFT/
        local_ORB/
        contours/
        image_EDGE/
        image_CORNER/
        image_BLOB/
        image_TEXTURE/

To skip a model or method, comment it out in the CONFIG section below.
Run with: uv run python e2e_test_all.py
"""

import sys
import os
import json
import time
import traceback
import gc

sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

import yolo_process
import image_processing
import embedding_process
import searching_similar
import get_yolo_pred_and_embeddings
import similarity_verification
from settings_and_utils import Settings


# ─────────────────────────────────────────────────────────────────────────────
# CONFIG — comment out anything you want to skip
# ─────────────────────────────────────────────────────────────────────────────

EMBEDDING_MODELS = [
    'siglip',       # google/siglip-so400m-patch14-384  1152D  (recommended for soil cores)
    'dino',         # facebook/dinov2-large             1024D
    'dino_giant',   # facebook/dinov2-giant             1536D  (large download ~3.5GB)
    'clip',         # openai/clip-vit-large-patch14     768D
    'radio',        # nvidia/RADIO via timm             768D   (large download)
    'vit_google',   # google/vit-base-patch16-224-in21k 768D
    # 'sd-vae',     # Stable Diffusion VAE — requires local finetuned_vae/ weights
]

DISTANCE_METHODS = [
    'euclidean',    # 1 / (1 + ||e1-e2||)  threshold: 0.55
    'cosine',       # dot(e1, e2)           threshold: 0.65
    'manhattan',    # 1 / (1 + L1(e1,e2))  threshold: 0.04
]

# Thresholds pulled from Settings — change there to affect all runs
DISTANCE_THRESHOLDS = {
    'euclidean': Settings.sim_tresh_euclidean,
    'cosine':    Settings.sim_tresh_cosine,
    'manhattan': Settings.sim_tresh_manhattan,
}

# Local feature matchers (verification_method='local_features')
LOCAL_FEATURE_METHODS = [
    'XFEAT',      # neural keypoints via torch.hub, CPU-friendly
    'LIGHTGLUE',  # LightGlue + SuperPoint transformer matcher
    'SIFT',       # classical OpenCV SIFT + FLANN
    'ORB',        # classical OpenCV ORB + BFMatcher
]

# Contour shape matching (verification_method='contours')
RUN_CONTOURS = True

# Image feature methods (verification_method='image_features')
IMAGE_FEATURE_METHODS = {
    'EDGE':    Settings.image_features_EDGE_similarity_threshold,
    'CORNER':  Settings.image_features_CORNER_similarity_threshold,
    'BLOB':    Settings.image_features_BLOB_similarity_threshold,
    'TEXTURE': Settings.image_features_TEXTURE_similarity_threshold,
}

OUTPUT_ROOT = 'e2e_results'

# ─────────────────────────────────────────────────────────────────────────────


def save_json(path, data):
    os.makedirs(os.path.dirname(path) or '.', exist_ok=True)
    with open(path, 'w', encoding='utf-8') as f:
        json.dump(data, f, indent=4, ensure_ascii=False)


def header(msg):
    bar = '=' * 64
    print(f"\n{bar}\n  {msg}\n{bar}")


def step(msg):
    print(f"\n  >> {msg}")


def run_verification(
    verif_key,
    verif_dir,
    predictions_path,
    similarity_path,
    verify_kwargs,
):
    """Run one verification pass, save JSON + collages. Returns (n_pairs, elapsed, error)."""
    verified_path = os.path.join(verif_dir, 'similarity_verified.json')
    collages_dir  = os.path.join(verif_dir, 'collages')
    os.makedirs(verif_dir, exist_ok=True)

    print(f"       {verif_key:<22}", end=' ', flush=True)
    t0 = time.time()
    try:
        verified = similarity_verification.verify_similarities(
            predictions_with_embeddings_path=predictions_path,
            sorted_pairs_path=similarity_path,
            formated_images_folder=Settings.formated_images_folder,
            verify_180_rotation=True,
            clean_debug_dir=True,
            **verify_kwargs,
        )
        save_json(verified_path, verified)
        if verified:
            image_processing.create_image_collages(
                collages_dir,
                verified_path,
                Settings.formated_images_folder,
                predictions_path,
            )
        elapsed = time.time() - t0
        print(f"{len(verified):>4} verified pairs  [{elapsed:.1f}s]")
        return len(verified), round(elapsed, 1), None
    except Exception as e:
        elapsed = time.time() - t0
        print(f"FAILED [{elapsed:.1f}s]  {e}")
        return 0, round(elapsed, 1), str(e)


def main():
    total_start = time.time()
    summary = {}
    os.makedirs(OUTPUT_ROOT, exist_ok=True)

    # ── Format images (once) ─────────────────────────────────────────
    header("1. Formatting images")
    image_processing.format_all_images_in_directory(
        Settings.input_folder,
        Settings.formated_images_folder,
        target_width=Settings.target_width,
        target_height=Settings.target_height,
    )

    # ── Load YOLO (once) ─────────────────────────────────────────────
    header("2. Loading YOLO model")
    yolo_model = yolo_process.init_yolo(Settings.yolo_model_path)
    if yolo_model is None:
        print("ERROR: Could not load YOLO model. Exiting.")
        return

    # ── Loop: embedding models ────────────────────────────────────────
    for emb_name in EMBEDDING_MODELS:
        header(f"Embedding model: {emb_name}")
        summary[emb_name] = {}

        emb_dir          = os.path.join(OUTPUT_ROOT, emb_name)
        predictions_path = os.path.join(emb_dir, 'predictions_with_embeddings.json')
        os.makedirs(emb_dir, exist_ok=True)

        # Load model
        step(f"Loading {emb_name}…")
        try:
            image_processor, emb_model, device = embedding_process.init_embedding(
                embedding_model_name=emb_name
            )
            if image_processor is None or (emb_model is None and device is not None):
                raise RuntimeError("init_embedding returned None — model weights may be missing")
        except Exception as e:
            print(f"  [SKIP] {e}")
            summary[emb_name]['_error'] = f"load: {e}"
            continue

        # Extract embeddings
        step("Extracting embeddings…")
        try:
            predictions = get_yolo_pred_and_embeddings.process_images_and_extract_embeddings(
                yolo_model, image_processor, emb_model, emb_name, device,
                Settings.formated_images_folder,
            )
            save_json(predictions_path, predictions)
            n_objects = sum(len(img.get('objects', [])) for img in predictions)
            print(f"     {len(predictions)} images, {n_objects} detected objects")
        except Exception as e:
            print(f"  [SKIP] Extraction failed: {e}")
            traceback.print_exc()
            summary[emb_name]['_error'] = f"extract: {e}"
            del image_processor, emb_model
            gc.collect()
            continue

        del image_processor, emb_model
        gc.collect()

        # ── Loop: distance methods ────────────────────────────────────
        for dist_method in DISTANCE_METHODS:
            step(f"Distance: {dist_method}  (threshold={DISTANCE_THRESHOLDS[dist_method]})")
            summary[emb_name][dist_method] = {}

            dist_dir        = os.path.join(emb_dir, dist_method)
            similarity_path = os.path.join(dist_dir, 'similarity.json')
            os.makedirs(dist_dir, exist_ok=True)

            # Compute candidate pairs
            try:
                loaded = searching_similar.load_and_preprocess_data(predictions_path)
                if not loaded:
                    raise RuntimeError("No valid objects with embeddings")
                pairs = searching_similar.find_most_similar_objects_diff_images(
                    objects=loaded,
                    sim_tresh=DISTANCE_THRESHOLDS[dist_method],
                    method=dist_method,
                    device='cpu',
                )
                save_json(similarity_path, pairs)
                print(f"     {len(pairs)} candidate pairs")
                del loaded
            except Exception as e:
                print(f"  [SKIP] Search failed: {e}")
                summary[emb_name][dist_method]['_error'] = f"search: {e}"
                continue

            print()

            # ── Local feature matchers ────────────────────────────────
            for lf_method in LOCAL_FEATURE_METHODS:
                verif_key = f"local_{lf_method}"
                verif_dir = os.path.join(dist_dir, verif_key)
                n, t, err = run_verification(
                    verif_key, verif_dir, predictions_path, similarity_path,
                    verify_kwargs=dict(
                        verification_method='local_features',
                        local_features_method=lf_method,
                        local_features_min_good_matches=Settings.local_features_min_good_matches,
                        local_features_ratio_thresh=Settings.local_features_ratio_thresh,
                        local_features_knnMatch_k=Settings.local_features_knnmatch_k,
                        local_features_flann_index_kdtree=Settings.local_features_flann_index_kdtree,
                        local_features_flann_index_kdtree_trees=Settings.local_features_flann_index_kdtree_trees,
                        local_features_search_params_checks=Settings.local_features_search_params_checks,
                        local_features_saving_good_matches_dir=os.path.join(verif_dir, 'debug_matches'),
                        local_features_check_dispersion=Settings.local_features_check_dispersion,
                        local_features_dispersion_grid_rows=Settings.local_features_dispersion_grid_rows,
                        local_features_dispersion_grid_cols=Settings.local_features_dispersion_grid_cols,
                        local_features_dispersion_min_occupied_cells_ratio=Settings.local_features_dispersion_min_occupied_cells_ratio,
                        xfeat_top_k=Settings.xfeat_top_k,
                        lightglue_extractor=Settings.lightglue_extractor,
                        lightglue_max_keypoints=Settings.lightglue_max_keypoints,
                        lightglue_confidence_threshold=Settings.lightglue_confidence_threshold,
                    ),
                )
                summary[emb_name][dist_method][verif_key] = {
                    'verified_pairs': n, 'elapsed_s': t,
                    'status': 'ok' if err is None else 'error',
                    **({'error': err} if err else {}),
                }

            # ── Contour shape matching ────────────────────────────────
            if RUN_CONTOURS:
                verif_key = 'contours'
                verif_dir = os.path.join(dist_dir, verif_key)
                n, t, err = run_verification(
                    verif_key, verif_dir, predictions_path, similarity_path,
                    verify_kwargs=dict(
                        verification_method='contours',
                        contour_binary_threshold_type=Settings.contour_binary_threshold_type,
                        contour_fixed_threshold_value=Settings.contour_fixed_threshold_value,
                        contour_adaptive_block_size=Settings.contour_adaptive_block_size,
                        contour_adaptive_C=Settings.contour_adaptive_C,
                        contour_match_shapes_method_str=Settings.contour_match_shapes_method_str,
                        contour_similarity_threshold=Settings.contour_similarity_threshold,
                        contour_min_contour_area_ratio=Settings.contour_min_contour_area_ratio,
                        saving_contour_debug_dir=os.path.join(verif_dir, 'debug_matches'),
                        contour_top_n_contours_to_compare=Settings.contour_top_n_contours_to_compare,
                    ),
                )
                summary[emb_name][dist_method][verif_key] = {
                    'verified_pairs': n, 'elapsed_s': t,
                    'status': 'ok' if err is None else 'error',
                    **({'error': err} if err else {}),
                }

            # ── Image feature methods ─────────────────────────────────
            for img_method, img_threshold in IMAGE_FEATURE_METHODS.items():
                verif_key = f"image_{img_method}"
                verif_dir = os.path.join(dist_dir, verif_key)
                n, t, err = run_verification(
                    verif_key, verif_dir, predictions_path, similarity_path,
                    verify_kwargs=dict(
                        verification_method='image_features',
                        image_features_method=img_method,
                        image_features_similarity_threshold=img_threshold,
                        image_features_debug_dir=os.path.join(verif_dir, 'debug_matches'),
                    ),
                )
                summary[emb_name][dist_method][verif_key] = {
                    'verified_pairs': n, 'elapsed_s': t,
                    'status': 'ok' if err is None else 'error',
                    **({'error': err} if err else {}),
                }

    # ── Print and save summary ────────────────────────────────────────
    header("SUMMARY")
    col_w = 24

    for emb_name, dist_results in summary.items():
        print(f"\n  [{emb_name}]")
        if '_error' in dist_results:
            print(f"    SKIPPED — {dist_results['_error']}")
            continue
        for dist_name, verif_results in dist_results.items():
            if dist_name.startswith('_'):
                continue
            if '_error' in verif_results:
                print(f"    {dist_name}: SKIPPED — {verif_results['_error']}")
                continue
            tokens = []
            for vk, vr in verif_results.items():
                if vr.get('status') == 'ok':
                    tokens.append(f"{vk}={vr['verified_pairs']}")
                else:
                    tokens.append(f"{vk}=ERR")
            print(f"    {dist_name:<12} {', '.join(tokens)}")

    total_elapsed = time.time() - total_start
    print(f"\n  Total time : {total_elapsed / 60:.1f} min")
    print(f"  Results    : {os.path.abspath(OUTPUT_ROOT)}/")

    summary_path = os.path.join(OUTPUT_ROOT, 'summary.json')
    save_json(summary_path, summary)
    print(f"  Summary    : {summary_path}\n")


if __name__ == '__main__':
    main()
