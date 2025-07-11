import glob
import os
from tqdm import tqdm
from PIL import Image
import numpy as np
import cv2

import yolo_process
import image_processing
import embedding_process


IMAGE_EXTENSIONS = ('*.jpg', '*.jpeg', '*.png')
CONFIDENCE_THRESHOLD = 0.4


def process_images_and_extract_embeddings(yolo_model, 
                                          image_processor, 
                                          embedding_model, 
                                          embedding_model_name,
                                          device, 
                                          image_folder, 
                                          extensions=IMAGE_EXTENSIONS, 
                                          conf_thresh=CONFIDENCE_THRESHOLD
                                          ) -> list:
    image_files = []
    for ext in extensions:
        image_files.extend(glob.glob(os.path.join(image_folder, ext)))
    if not image_files:
        print(f"No images found in {image_folder} with extensions {extensions}")
        return
    print(f"Found {len(image_files)} images to process.")

    all_predictions_data = []

    for image_path in tqdm(image_files, desc="Processing images"):
        image_name = os.path.basename(image_path)
        image_entry = {
            "image_name": image_name,
            "objects": []
        }

        try:
            try:
                pil_image = Image.open(image_path).convert("RGB")
            except Exception as e_pil:
                print(f"Error: Could not open image {image_path} with PIL. Error: {e_pil}. Skipping.")
                image_entry["error"] = f"PIL could not open image: {e_pil}"
                all_predictions_data.append(image_entry)
                continue
            try:
                img_bytes = np.fromfile(image_path, np.uint8)
                cv_image = cv2.imdecode(img_bytes, cv2.IMREAD_COLOR)
                if cv_image is None: # if imdecode also fails
                    raise ValueError("imdecode returned None")
            except Exception as e_cv_fallback:
                print(f"Warning: Could not read image {image_path} with OpenCV (standard or fallback). Error: {e_cv_fallback}. Skipping.")
                image_entry["error"] = f"OpenCV could not read image: {e_cv_fallback}"
                all_predictions_data.append(image_entry)
                continue
            
            yolo_process.yolo_predict(yolo_model, pil_image, image_entry, conf_thresh)
            for i in range(len(image_entry['objects'])):
                try:
                    obb_flat_coords = image_entry['objects'][i]['coords']
                    cropped_obb_cv = image_processing.crop_obb_from_image(cv_image, obb_flat_coords)
                except Exception as e:
                    print(f"Error during crop object, image name {image_name}: {e}")
                    continue
                try:
                    embedding = embedding_process.get_embedding(
                        image_processor, 
                        embedding_model, 
                        cropped_obb_cv, 
                        device=device, 
                        model_name_hint=embedding_model_name
                    )
                except Exception as e:
                    print(f"Error during embedding extraction for an object in {image_name}: {e}")
                    embedding = []
                image_entry['objects'][i]['embedding'] = embedding
                image_entry['objects'][i]['coords'] = image_entry['objects'][i]['coords'].tolist()
            
        except Exception as e:
            print(f"Error processing {image_name}: {e}")
            import traceback
            traceback.print_exc() # This will print the full traceback
            image_entry["error"] = str(e)

        all_predictions_data.append(image_entry)
    
    return all_predictions_data