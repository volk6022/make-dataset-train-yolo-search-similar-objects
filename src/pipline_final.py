import json
import time

import yolo_process
import image_processing
import embedding_process
import searching_similar
import get_yolo_pred_and_embeddings
from settings_and_utils import Settings
import similarity_verification


def save_json(output_file_path, data):
    with open(output_file_path, 'w', encoding='utf-8') as f:
            json.dump(data, f, indent=4, ensure_ascii=False)


if __name__ == "__main__":
    start = time.time()

    image_processing.format_all_images_in_directory(
        Settings.input_folder, 
        Settings.formated_images_folder, 
        target_width=Settings.target_width, 
        target_height=Settings.target_height
    )

    yolo_model = yolo_process.init_yolo(Settings.yolo_model_path)
    if yolo_model is None:
        exit()
    image_processor, embedding_model, device = embedding_process.init_embedding(
        embedding_model_name=Settings.embedding_model_name
    )
    if image_processor is None or embedding_model is None and device is not None:
        exit()

    predictions_with_embeddings = get_yolo_pred_and_embeddings.process_images_and_extract_embeddings(
        yolo_model, 
        image_processor, 
        embedding_model, 
        Settings.embedding_model_name, 
        device, 
        Settings.formated_images_folder
    )

    save_json(Settings.output_predictions_json_file, predictions_with_embeddings)
    del yolo_model, predictions_with_embeddings, image_processor, embedding_model, device
    
    loaded_json = searching_similar.load_and_preprocess_data(Settings.output_predictions_json_file)
    sorted_pairs = searching_similar.find_most_similar_objects_diff_images(
        objects=loaded_json, 
        sim_tresh=Settings.sim_tresh_euclidean,
        method="euclidean",
        device='cpu'
    )
    save_json(Settings.output_similarity_json_file, sorted_pairs)
    del loaded_json, sorted_pairs

    verified_pairs_local_features = similarity_verification.verify_similarities(
        predictions_with_embeddings_path=Settings.output_predictions_json_file,
        sorted_pairs_path=Settings.output_similarity_json_file,
        formated_images_folder=Settings.formated_images_folder,
        verification_method=Settings.local_features_verification_method,
        verify_180_rotation=True,

        local_features_min_good_matches=Settings.local_features_min_good_matches,
        local_features_ratio_thresh=Settings.local_features_ratio_thresh,
        local_features_method=Settings.local_features_method,
        local_features_knnMatch_k=Settings.local_features_knnmatch_k,
        local_features_flann_index_kdtree=Settings.local_features_flann_index_kdtree,
        local_features_flann_index_kdtree_trees=Settings.local_features_flann_index_kdtree_trees,
        local_features_search_params_checks=Settings.local_features_search_params_checks,
        local_features_saving_good_matches_dir=Settings.local_features_saving_good_matches_dir,
        local_features_check_dispersion=Settings.local_features_check_dispersion,
        local_features_dispersion_grid_cols=Settings.local_features_dispersion_grid_cols,
        local_features_dispersion_grid_rows=Settings.local_features_dispersion_grid_rows,
        local_features_dispersion_min_occupied_cells_ratio=Settings.local_features_dispersion_min_occupied_cells_ratio
    )

    verified_pairs_image_features = similarity_verification.verify_similarities(
        predictions_with_embeddings_path=Settings.output_predictions_json_file,
        sorted_pairs_path=Settings.output_similarity_json_file,
        formated_images_folder=Settings.formated_images_folder,
        verification_method=Settings.image_features_verification_method,
        verify_180_rotation=True,
        image_features_debug_dir=Settings.image_features_EDGE_debug_dir,
        image_features_method=Settings.image_features_EDGE_method,
        image_features_similarity_threshold=Settings.image_features_EDGE_similarity_threshold
    )

    verified_pairs = []
    verified_pairs.extend(verified_pairs_local_features)
    verified_ids = [x["pair_id"] for x in verified_pairs]
    for i in range(len(verified_pairs_image_features)):
        if verified_pairs_image_features[i]["pair_id"] not in verified_ids:
            verified_ids.append(verified_pairs_image_features[i]["pair_id"])
            verified_pairs.append(verified_pairs_image_features[i])
    sorted_pairs = sorted(verified_pairs, key=lambda x: x["similarity"], reverse=True)
    save_json(Settings.output_similarity_json_file_verified, sorted_pairs)
    del verified_pairs


    image_processing.create_image_collages(
        Settings.output_collages_dir, 
        Settings.output_similarity_json_file_verified,
        Settings.formated_images_folder, 
        Settings.output_predictions_json_file
    )

    print('\ntotal time:', time.time() - start)
