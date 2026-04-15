import os
import shutil


class Settings:
    # formatimng settings
    input_folder = 'embedding_test_images/'
    formated_images_folder = 'pipline_test_formated/'
    target_width = 640
    target_height = 640

    # yolo and embedding settings
    yolo_model_path = 'runs/obb/120/train/weights/best.pt'
    output_predictions_json_file = 'predictions_with_embeddings.json'
    # embedding_model_name options: 'dino', 'dino_giant', 'clip', 'siglip', 'radio', 'vit_google', 'sd-vae'
    # siglip recommended for soil cores: 384px input captures layered texture detail
    embedding_model_name = 'siglip'

    # embeddings comparing settings
    output_similarity_json_file = 'similarity.json'
    sim_tresh_cosine = 0.65
    sim_tresh_euclidean = 0.55
    sim_tresh_manhattan = 0.04
    # Cap on candidate pairs sent to any verifier (sorted by similarity desc).
    # Prevents runaway runtimes when a loose threshold produces huge candidate sets.
    # Set to None to disable.
    max_candidate_pairs_to_verify = 5000

    # semilarity verification settings
    # local features verication method settings
    local_features_verification_method = 'local_features'
    local_features_min_good_matches = 20
    local_features_ratio_thresh = 0.8
    # local_features_method options: "SIFT", "ORB", "XFEAT", "LIGHTGLUE"
    local_features_method = "XFEAT"
    local_features_knnmatch_k = 2
    local_features_flann_index_kdtree = 1
    local_features_flann_index_kdtree_trees = 20
    local_features_search_params_checks = 100
    local_features_saving_good_matches_dir = 'local_features_verification_good_matches/'
    local_features_check_dispersion=True # Enable/disable dispersion check
    local_features_dispersion_grid_rows=10
    local_features_dispersion_grid_cols=2
    local_features_dispersion_min_occupied_cells_ratio=0.7
    # XFeat settings (used when local_features_method == 'XFEAT')
    xfeat_top_k = 4096
    # LightGlue settings (used when local_features_method == 'LIGHTGLUE')
    lightglue_extractor = 'superpoint'  # 'superpoint', 'disk', 'aliked'
    lightglue_max_keypoints = 2048
    lightglue_confidence_threshold = 0.5
    # contour verification settings
    contour_verification_method = 'contours'
    contour_binary_threshold_type="adaptive" # "otsu", "adaptive", "fixed"
    contour_fixed_threshold_value=63
    contour_adaptive_block_size=11 # Should be odd
    contour_adaptive_C=2
    contour_match_shapes_method_str="I2" # "I1", "I2", "I3"
    contour_similarity_threshold=0.5 # Lower is better for matchShapes
    contour_min_contour_area_ratio=0.01 # Min contour area relative to image area
    contour_saving_debug_dir='contour_verification_good_matches/'
    contour_top_n_contours_to_compare=10
    # image features verification settings
    image_features_verification_method = 'image_features'
    # EDGE
    image_features_EDGE_method = 'EDGE'
    # image_features_EDGE_similarity_threshold = 0.13
    image_features_EDGE_similarity_threshold = 0.2
    image_features_EDGE_debug_dir = 'image_features_good_matches/'
    # EDGE
    image_features_CORNER_method = 'CORNER'
    # image_features_CORNER_similarity_threshold = 0.5
    image_features_CORNER_similarity_threshold = 0.2
    image_features_CORNER_debug_dir = 'image_features_good_matches/'
    # BLOB
    image_features_BLOB_method = 'BLOB'
    image_features_BLOB_similarity_threshold = 0.65
    image_features_BLOB_debug_dir = 'image_features_good_matches/'
    # TEXTURE
    image_features_TEXTURE_method = 'TEXTURE'
    image_features_TEXTURE_similarity_threshold = 0.1
    image_features_TEXTURE_debug_dir = 'image_features_good_matches/'

    output_similarity_json_file_verified = 'similarity_verified.json'

    # collage settings
    output_collages_dir = 'pipline_test_collage/'



def delete_directory_and_contents(directory_path):
    """
    Deletes the specified directory and ALL its contents (files and subdirectories).
    """
    if not os.path.isdir(directory_path):
        # print(f"Error: Directory '{directory_path}' not found.")
        return

    try:
        shutil.rmtree(directory_path)
        print(f"Successfully deleted directory and all its contents: {directory_path}")
    except Exception as e:
        print(f"Error deleting directory {directory_path}: {e}")