import os
import json
from collections import defaultdict
from tqdm import tqdm
import numpy as np
import cv2

from image_comparator import ImageComparator, create_similarity_report_with_images
import image_processing
from settings_and_utils import Settings



def crop_and_save_croped_objects(
        predictions_with_embeddings_path,
        sorted_pairs_path,
        formated_images_folder,
        folder_to_save_croped
    ):
    os.makedirs(folder_to_save_croped, exist_ok=True)
    with open(predictions_with_embeddings_path, 'r', encoding='utf-8') as f:
        predictions_with_embeddings = json.load(f)
    with open(sorted_pairs_path, 'r', encoding='utf-8') as f:
        sorted_pairs = json.load(f)

    object_lookup = {}
    for img_data in predictions_with_embeddings:
        if "error" in img_data:
            continue
        object_lookup[img_data["image_name"]] = {
            obj["id"]: obj for obj in img_data.get("objects", [])
        }

    print("Preprocessing: Identifying and cropping objects...")

    unique_objects_to_crop = set()
    for pair in sorted_pairs:
        obj1_info, obj2_info = pair["object1"], pair["object2"]
        unique_objects_to_crop.add((obj1_info["image_name"], obj1_info["id"]))
        unique_objects_to_crop.add((obj2_info["image_name"], obj2_info["id"]))

    objects_by_image_name = defaultdict(list)
    for img_name, obj_id in unique_objects_to_crop:
        objects_by_image_name[img_name].append(obj_id)

    cropped_object_cache = {}
    pbar_preprocessing = tqdm(objects_by_image_name.items(), desc="Loading images & cropping objects")
    num_successfully_saved = 0
    num_failed_save = 0

    for image_name, obj_ids_in_image in pbar_preprocessing:
        img_path = os.path.join(formated_images_folder, image_name)
        cv_image_full = None
        try:
            # Use np.fromfile for paths with non-ASCII characters, then imdecode
            img_bytes = np.fromfile(img_path, np.uint8)
            cv_image_full = cv2.imdecode(img_bytes, cv2.IMREAD_COLOR)
            if cv_image_full is None:
                print(f"Warning: Could not read image {img_path}. Objects from this image will be skipped.")
                for obj_id in obj_ids_in_image:
                    cropped_object_cache[(image_name, obj_id)] = None
                continue
        except Exception as e:
            print(f"Error reading image {img_path}: {e}. Objects will be skipped.")
            for obj_id in obj_ids_in_image:
                cropped_object_cache[(image_name, obj_id)] = None
            continue

        for obj_id in obj_ids_in_image:
            object_key = (image_name, obj_id)
            orig_obj_data = object_lookup.get(image_name, {}).get(obj_id)
            if not orig_obj_data:
                print(f"Warning: No original data for {object_key}.")
                cropped_object_cache[object_key] = None
                continue

            coords_np = np.array(orig_obj_data["coords"])
            cropped_obj_cv = None # Initialize
            try:
                cropped_obj_cv = image_processing.crop_obb_from_image(cv_image_full, coords_np)
                cropped_object_cache[object_key] = cropped_obj_cv # Store in cache

                # --- ADDED SAVE LOGIC ---
                if cropped_obj_cv is not None:
                    # Create a unique filename for the cropped object
                    # Remove original extension from image_name and add object_id
                    base_name, _ = os.path.splitext(image_name)
                    save_filename = f"{base_name}_obj_{obj_id}.png" # Using PNG, can change to .jpg
                    save_path = os.path.join(folder_to_save_croped, save_filename)

                    try:
                        # cv2.imwrite cannot handle non-ASCII paths directly on all systems
                        # A common workaround is to use imencode and write bytes
                        is_success, im_buf_arr = cv2.imencode(".png", cropped_obj_cv)
                        if is_success:
                            im_buf_arr.tofile(save_path)
                            num_successfully_saved += 1
                            # print(f"Successfully saved: {save_path}") # Optional: for verbose logging
                        else:
                            print(f"Error: cv2.imencode failed for {save_path}")
                            num_failed_save += 1
                    except Exception as e_save:
                        print(f"Error saving cropped image {save_path}: {e_save}")
                        num_failed_save +=1
                # --- END ADDED SAVE LOGIC ---

            except Exception as e_crop:
                print(f"Error cropping {object_key}: {e_crop}")
                cropped_object_cache[object_key] = None # Mark as failed in cache

    num_total_unique_objects = len(unique_objects_to_crop)
    num_successfully_cropped_in_cache = sum(1 for obj_cv in cropped_object_cache.values() if obj_cv is not None)
    num_failed_crop_or_load = num_total_unique_objects - num_successfully_cropped_in_cache

    print(f"\nPreprocessing finished. Attempted to process: {num_total_unique_objects} unique objects.")
    print(f"Successfully cropped (and cached in memory): {num_successfully_cropped_in_cache}.")
    print(f"Failed to load image or crop object: {num_failed_crop_or_load}.")
    print(f"Successfully saved to disk: {num_successfully_saved} cropped images.")
    if num_failed_save > 0:
        print(f"Failed to save to disk (despite successful crop): {num_failed_save} cropped images.")


def demo_image_comparison():
    """
    Demonstration function showing how to use the ImageComparator.
    
    Note: You'll need to provide actual image paths for this to work.
    """
    
    # Example usage (replace with actual image paths)
    reference_image = "cropped_objects/skv.266_керн1-5_obj_4.png"  # Replace with your reference image path
    comparison_images = [
        "cropped_objects/_skv.266_керн1-5_obj_4_rotated_180.png",
        "cropped_objects/_skv.266_керн1-5_obj_4_rotated_-90.png",  # Replace with your image paths
        "cropped_objects/_skv.266_керн1-5_obj_4_rotated_+90.png",
        "cropped_objects/skv.266_керн1-5_obj_4.png",
        "cropped_objects/skv.252_керн6-10_obj_0.png"
    ]
    
    try:
        # Initialize comparator
        comparator = ImageComparator(reference_image)
        
        # Compare multiple images
        results_df = comparator.compare_multiple_images(comparison_images)
        
        # Display results
        print("Similarity Comparison Results:")
        print(results_df.to_string(index=False, float_format='%.4f'))
        
        # Generate summary report
        comparator.generate_summary_report(results_df)
        
        # Create and display heatmap
        # comparator.create_similarity_heatmap(results_df, 'similarity_heatmap.png')
        
        # Save results to CSV
        results_df.to_csv('image_similarity_results.csv', index=False)
        print("\nResults saved to 'image_similarity_results.csv'")

        create_similarity_report_with_images('image_similarity_results.csv', "cropped_objects/")
        
        # return results_df
        
    except Exception as e:
        print(f"{e}")

# If running this script directly
if __name__ == "__main__":
    # crop_and_save_croped_objects(
    #     Settings.output_predictions_json_file,
    #     Settings.output_similarity_json_file,
    #     Settings.formated_images_folder,
    #     'cropped_objects/'
    # )
    
    demo_image_comparison()