import numpy as np
import cv2
# from PIL import ExifTags
import os
from tqdm import tqdm
from PIL import Image, ImageDraw
import json

from settings_and_utils import delete_directory_and_contents



def crop_obb_from_image(image_cv, obb_coords_xyxyxyxy):
    """
    Crops an Oriented Bounding Box (OBB) from an image using OpenCV.
    The output cropped image will be vertically oriented (i.e., height >= width).
    The OBB coordinates are expected as a flat list/array [x0,y0, x1,y1, x2,y2, x3,y3],
    representing the four corners P0, P1, P2, P3 in a consistent order (e.g., clockwise).
    """

    try:
        src_pts_original_order = obb_coords_xyxyxyxy.reshape((4, 2)).astype(np.float32)
    except ValueError as e:
        return None
    
    vec_side1 = src_pts_original_order[0] - src_pts_original_order[1]
    vec_side2 = src_pts_original_order[1] - src_pts_original_order[2]
    side1_length_f = float(np.linalg.norm(vec_side1))
    side2_length_f = float(np.linalg.norm(vec_side2))
    transform_src_pts = src_pts_original_order

    if side1_length_f > side2_length_f:
        output_w_f = side2_length_f
        output_h_f = side1_length_f
        transform_src_pts = np.array([
            src_pts_original_order[1],  # P1
            src_pts_original_order[2],  # P2
            src_pts_original_order[3],  # P3
            src_pts_original_order[0]   # P0
        ], dtype=np.float32)
    else:
        output_w_f = side1_length_f
        output_h_f = side2_length_f

    output_w = int(round(output_w_f))
    output_h = int(round(output_h_f))

    if output_w < 1 or output_h < 1:
        return None
    
    dst_pts = np.array([
        [0, 0],                        # Top-Left
        [output_w - 1, 0],             # Top-Right
        [output_w - 1, output_h - 1],  # Bottom-Right
        [0, output_h - 1]              # Bottom-Left
    ], dtype=np.float32)

    try:
        # Get the perspective transformation matrix
        M = cv2.getPerspectiveTransform(transform_src_pts, dst_pts)
        
        # Perform the perspective warp. dsize for warpPerspective is (width, height).
        warped_crop = cv2.warpPerspective(image_cv, M, (output_w, output_h))
    except cv2.error as e:
        return None
        
    return warped_crop

### image preprocess

def process_single_image(image_path, 
                         output_path, 
                         target_width, 
                         target_height, 
                         fixed_rotation=0, 
                         quality=85, 
                         skip_exif=True,
                         ):
    """
    Processes a single image: corrects orientation, applies fixed rotation, resizes, and saves.
    Returns True on success, False on failure.
    Optionally displays the processed image in the notebook.
    """
    try:
        img = Image.open(image_path)
        original_filename = os.path.basename(image_path)
        print(f"Processing {original_filename}...")
        print(f"  Resizing to {target_width}x{target_height}...")
        img_resized = img.resize((target_width, target_height), Image.Resampling.LANCZOS)
        base, ext = os.path.splitext(output_path)
        save_img = img_resized
        if ext.lower() in ['.jpg', '.jpeg']:
            if save_img.mode in ['RGBA', 'P']:
                print(f"  Converting image mode from {save_img.mode} to RGB for JPEG saving.")
                save_img = save_img.convert('RGB')
            save_img.save(output_path, quality=quality)
        else:
            save_img.save(output_path)

        print(f"  Saved processed image to: {output_path}")
        return True

    except FileNotFoundError:
        print(f"Error: Image file not found at {image_path}")
    except IOError as e:
        print(f"Error: Could not open/read image: {image_path}. It might be corrupted or not a valid image. Details: {e}")
    except Exception as e:
        print(f"An unexpected error occurred while processing {image_path}: {e}")
    return False

def format_all_images_in_directory(input_dir, 
                                   output_dir, 
                                   target_width=640, 
                                   target_height=640, 
                                   fixed_rotation_angle=0, 
                                   jpeg_quality=90,
                                   skip_exif_correction=True
                                   ):
    delete_directory_and_contents(output_dir)
    os.makedirs(output_dir, exist_ok=True)
    image_extensions = ('.png', '.jpg', '.jpeg')
    processed_count = 0
    failed_count = 0

    image_files = [f for f in os.listdir(input_dir) if f.lower().endswith(image_extensions)]
    total_images = len(image_files)

    with tqdm(total=total_images, desc="Formating Images") as pbar:
        for filename in image_files:
            input_path = os.path.join(input_dir, filename)
            output_filename = filename
            output_path = os.path.join(output_dir, output_filename)

            try:
                img = Image.open(input_path)
                img_resized = img.resize((target_width, target_height), Image.Resampling.LANCZOS)

                base, ext = os.path.splitext(output_path)
                save_img = img_resized
                if ext.lower() in ['.jpg', '.jpeg']:
                    if save_img.mode in ['RGBA', 'P']:
                        save_img = save_img.convert('RGB')
                    save_img.save(output_path, quality=jpeg_quality)
                else:
                    save_img.save(output_path)
                processed_count += 1
            except Exception:
                failed_count += 1
            pbar.update(1)

    print(f"Successfully processed: {processed_count} image(s).")
    print(f"Failed to process: {failed_count} image(s).")
    print(f"Output saved in: {os.path.abspath(output_dir)}")


### make collage
PADDING = 10
BACKGROUND_COLOR = (255, 255, 255)
BOX_COLOR = "red"  # Or (255, 0, 0)
BOX_WIDTH = 3

def load_annotations(annotations_json_path):
    """
    Loads annotations and structures them for easy lookup.
    Returns a dictionary:
    {
        "image_name1.jpg": {
            "id1": [[x1,y1], [x2,y2], ...],
            "id2": [[x1,y1], [x2,y2], ...]
        },
        "image_name2.jpg": { ... }
    }
    """
    try:
        with open(annotations_json_path, 'r', encoding='utf-8') as f:
            annotation_data_list = json.load(f)
    except FileNotFoundError:
        print(f"Error: Annotations JSON file '{annotations_json_path}' not found.")
        return None
    except json.JSONDecodeError as e:
        print(f"Error: Could not decode JSON from '{annotations_json_path}'. Details: {e}")
        return None

    if not isinstance(annotation_data_list, list):
        print(f"Error: Expected a list of objects in annotations JSON, but got {type(annotation_data_list)}.")
        return None

    processed_annotations = {}
    for item in annotation_data_list:
        img_name = item.get('image_name')
        objects = item.get('objects')
        if not img_name or not isinstance(objects, list):
            print(f"Warning: Skipping invalid annotation item: {item}")
            continue
        
        processed_annotations[img_name] = {}
        for obj_data in objects:
            obj_id = obj_data.get('id')
            coords = obj_data.get('coords')
            if obj_id is not None and coords: # Ensure id is not None (0 is a valid ID)
                # Convert coords to list of tuples if they aren't already for PIL
                pil_coords = [tuple(p) for p in coords]
                processed_annotations[img_name][obj_id] = pil_coords
            else:
                print(f"Warning: Skipping object with missing id or coords in {img_name}: {obj_data}")
    return processed_annotations


def draw_bounding_box(image, coordinates, color="red", width=3):
    """Draws a bounding box on the image."""
    if not coordinates:
        return image
    draw = ImageDraw.Draw(image)
    # Coordinates should be a list of (x,y) tuples for polygon
    # e.g., [(x1,y1), (x2,y2), (x3,y3), (x4,y4)]
    draw.polygon(coordinates, outline=color, width=width)
    return image


def create_image_collages(output_collages_dir, json_file_path, input_images_dir, annotations_json_path, clean_existing_dir=True, max_collages=1000):
    """
    Reads image pairs from a JSON file, draws bounding boxes from an annotations file,
    and creates side-by-side collages with filename format: XXXX_collade_ID1_vs_ID2.jpg

    max_collages: cap on how many collages to write (top-N by list order, which is
                  descending similarity score). Set to None for no limit.
    """
    # 1. Ensure output directory exists
    if clean_existing_dir:
        delete_directory_and_contents(output_collages_dir)
    os.makedirs(output_collages_dir, exist_ok=True)

    # 2. Load JSON data for pairs
    try:
        with open(json_file_path, 'r', encoding='utf-8') as f:
            pairs_data = json.load(f)
    except FileNotFoundError:
        print(f"Error: Pairs JSON file '{json_file_path}' not found.")
        return
    except json.JSONDecodeError as e:
        print(f"Error: Could not decode JSON from '{json_file_path}'. Details: {e}")
        return

    if not isinstance(pairs_data, list):
        print(f"Error: Expected a list of objects in pairs JSON, but got {type(pairs_data)}.")
        return
    
    if not pairs_data:
        print("No structurally valid image pairs found in the JSON to create collages.")
        return

    if max_collages is not None and len(pairs_data) > max_collages:
        print(f"Limiting collages to top {max_collages} of {len(pairs_data)} pairs.")
        pairs_data = pairs_data[:max_collages]

    potential_collages_count = len(pairs_data)

    # Load annotations
    annotations = load_annotations(annotations_json_path)
    if annotations is None:
        print("Could not load annotations. Proceeding without drawing boxes.")
        # To strictly require annotations, you might want to `return` here.
        # For now, we'll allow it to proceed and just skip drawing if annotations are missing.
        annotations = {} # Ensure it's an empty dict if loading failed but we proceed

    num_digits_for_sequence = len(str(potential_collages_count)) if potential_collages_count > 0 else 1

    # 3. Process each pair in the JSON data
    for i, pair_info in tqdm(enumerate(pairs_data), total=len(pairs_data), desc='Creating collages'):
        try:
            obj1_info = pair_info.get('object1')
            obj2_info = pair_info.get('object2')

            if not obj1_info or not obj2_info:
                print(f"Warning: Skipping item at index {i} due to missing 'object1' or 'object2'.")
                continue

            img_name1 = obj1_info.get('image_name')
            cmp_id1 = obj1_info.get('id') # Expecting this to be the ID for annotation lookup
            img_name2 = obj2_info.get('image_name')
            cmp_id2 = obj2_info.get('id') # Expecting this to be the ID for annotation lookup

            # Fallback IDs for filename if actual IDs are missing from pair_info (not for annotation lookup)
            filename_cmp_id1 = cmp_id1 if cmp_id1 is not None else f"obj1_idx{i}"
            filename_cmp_id2 = cmp_id2 if cmp_id2 is not None else f"obj2_idx{i}"


            if not img_name1 or not img_name2:
                print(f"Warning: Skipping pair with (filename) IDs {filename_cmp_id1}-{filename_cmp_id2} due to missing image_name.")
                continue
            
            if cmp_id1 is None:
                print(f"Warning: Missing 'id' for object1 in pair {i} (image: {img_name1}). Cannot draw box.")
            if cmp_id2 is None:
                print(f"Warning: Missing 'id' for object2 in pair {i} (image: {img_name2}). Cannot draw box.")


            full_img_path1 = os.path.join(input_images_dir, img_name1)
            full_img_path2 = os.path.join(input_images_dir, img_name2)

            # 4. Open the two images
            try:
                image1 = Image.open(full_img_path1)
                image2 = Image.open(full_img_path2)
            except FileNotFoundError as e:
                print(f"Error: Could not open image: {e}. Skipping pair {filename_cmp_id1}-{filename_cmp_id2}.")
                continue
            
            image1 = image1.convert('RGB')
            image2 = image2.convert('RGB')

            # --- Draw bounding boxes ---
            coords1 = None
            if img_name1 in annotations and cmp_id1 is not None and cmp_id1 in annotations[img_name1]:
                coords1 = annotations[img_name1][cmp_id1]
                image1 = draw_bounding_box(image1.copy(), coords1, BOX_COLOR, BOX_WIDTH) # Draw on a copy
            else:
                if cmp_id1 is not None: # Only warn if an ID was provided but not found
                    print(f"Warning: No annotation found for image '{img_name1}' with ID '{cmp_id1}'.")

            coords2 = None
            if img_name2 in annotations and cmp_id2 is not None and cmp_id2 in annotations[img_name2]:
                coords2 = annotations[img_name2][cmp_id2]
                image2 = draw_bounding_box(image2.copy(), coords2, BOX_COLOR, BOX_WIDTH) # Draw on a copy
            else:
                if cmp_id2 is not None: # Only warn if an ID was provided but not found
                    print(f"Warning: No annotation found for image '{img_name2}' with ID '{cmp_id2}'.")
            # --- End drawing ---

            # 5. Create the collage
            width1, height1 = image1.size
            width2, height2 = image2.size

            collage_width = width1 + width2 + PADDING
            collage_height = max(height1, height2)

            collage_image = Image.new('RGB', (collage_width, collage_height), BACKGROUND_COLOR)
            
            # Vertical centering (optional, current code is top-align)
            y1_offset = (collage_height - height1) // 2
            y2_offset = (collage_height - height2) // 2
            
            collage_image.paste(image1, (0, y1_offset))
            collage_image.paste(image2, (width1 + PADDING, y2_offset))

            # 6. Save the collage
            sequence_str = str(i).zfill(num_digits_for_sequence)
            
            # Ensure IDs are strings for filename
            str_cmp_id1 = str(filename_cmp_id1).replace(" ", "_") # Sanitize spaces
            str_cmp_id2 = str(filename_cmp_id2).replace(" ", "_")

            output_filename = f"{sequence_str}_collage_{img_name1}_obj_{str_cmp_id1}_vs_{img_name2}_obj_{str_cmp_id2}.jpg"
            output_path = os.path.join(output_collages_dir, output_filename)
            collage_image.save(output_path)

        except Exception as e:
            print(f"An unexpected error occurred while processing item at index {i}: {pair_info}. Error: {e}")
            import traceback
            traceback.print_exc()


    print("\nCollage creation process finished.")
