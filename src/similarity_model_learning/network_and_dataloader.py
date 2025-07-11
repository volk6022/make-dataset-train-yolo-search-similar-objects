import torch
import torch.nn as nn
from torch.utils.data import Dataset
import numpy as np
import json
import os
from tqdm import tqdm
import cv2
import sys

sys.path.append("C:/Users/bhunp/python312/Scripts/local_projects/search_for_not_unique_kerns/src")

import embedding_process
import image_processing


image_extensions = ('.png', '.jpg', '.jpeg')

def place_embedding_on_device(emb_list: list, device: torch.device):
    emb = torch.tensor(emb_list, device=device, dtype=torch.float32)
    norm = torch.linalg.norm(emb)
    if norm > 1e-9: # Safeguard against near-zero norms
        emb = emb / norm
    else:
        emb = torch.zeros_like(emb)
    return emb

def place_pair_embeddings_on_device(obj_pair: dict, device: torch.device):
    obj1_emb = obj_pair['object1']['embedding']
    obj2_emb = obj_pair['object2']['embedding']
    obj_pair['object1']['embedding'] = place_embedding_on_device(obj1_emb, device)
    obj_pair['object2']['embedding'] = place_embedding_on_device(obj2_emb, device)

class SiamesePairedDataset(Dataset):
    def __init__(self, similarity_json: str | list, verified_collages: str | list, device='cuda', max_verified_labels: int = None):
        self.device = device
        self.object_pairs = []
        if isinstance(similarity_json, str):
            try:
                with open(similarity_json, 'r', encoding='utf-8') as f:
                    self.object_pairs = json.load(f)
            except FileNotFoundError:
                print(f"Error: JSON file not found at {similarity_json}")
            except json.JSONDecodeError:
                print(f"Error: Could not decode JSON from {similarity_json}")
        elif isinstance(similarity_json, list):
            self.object_pairs = similarity_json
        if max_verified_labels:
            self.object_pairs = self.object_pairs[:max_verified_labels+1]
        for pair in self.object_pairs:
            place_pair_embeddings_on_device(pair, self.device)

        if isinstance(verified_collages, str):
            self.verified_collages_files = [f for f in os.listdir(verified_collages) if f.lower().endswith(image_extensions)]
            self.verified_collages_ids = list(map(
                lambda x: int(x[:x.find('_')]), 
                self.verified_collages_files
            ))
            self.verified_collages_labels = [
                torch.tensor(1.0, device=device, dtype=torch.float32) if i in self.verified_collages_ids else torch.tensor(0.0, device=device, dtype=torch.float32) 
                for i in range(len(self.object_pairs))
            ]
        elif isinstance(verified_collages, list):
            self.verified_collages_labels = verified_collages
        self.num_ftrs = len(self.object_pairs[0]['object1']['embedding'])

    def __len__(self):
        return len(self.object_pairs)

    def __getitem__(self, idx):
        emb1 = self.object_pairs[idx]['object1']['embedding']
        emb2 = self.object_pairs[idx]['object2']['embedding']
        return [emb1, emb2], self.verified_collages_labels[idx]


class SiamesePairedDataset_transform(Dataset):
    def __init__(self, 
            model_name: str = 'sd-vae', 
            device: str | torch.device = 'cuda', 
            max_verified_labels: int | None = None,
            transforms = None,
            load_embedding_model = True,
        ):
        self.model_name = model_name
        self.device = device
        self.max_verified_labels = max_verified_labels
        self.transforms = transforms
        self.load_embedding_model = load_embedding_model
        if self.load_embedding_model:
            self.image_processor, self.embedding_model, self.device = embedding_process.init_embedding(
                embedding_model_name=self.model_name,
                device=self.device
            )
        self.cropped_obj_cv = {
            # "<image_name>_obj_0": <cv_image>
        }
        self.obj_embeddings = {
            # "<image_name>_obj_0": list | torch.Tensor
        }
        self.pair_obj_names = [
            # ("<image_name>_obj_0", "<image_name>_obj_1")
        ]
        self.siamese_paire_label = [
            # torch.tensor(0.0, device=device, dtype=torch.float32) or torch.tensor(1.0, device=device, dtype=torch.float32)
        ]

        self.num_ftrs = 0
        self._transforms_applied = False

    def __len__(self):
        return len(self.pair_obj_names)

    def __getitem__(self, idx):
        obj_name_1, obj_name_2 = self.pair_obj_names[idx]
        # if obj_name_1 not in self.obj_embeddings or obj_name_2 not in self.obj_embeddings:
        #     raise RuntimeError(f"Embeddings not found for pair ({obj_name_1}, {obj_name_2}). "
        #                        "Ensure _calc_all_objects_embeddings() was called after loading data.")
        return [self.obj_embeddings[obj_name_1], self.obj_embeddings[obj_name_2]], self.siamese_paire_label[idx]

    def _get_or_crop_object(self, 
                            image_name: str, 
                            obj_id: int, 
                            objects_data_map_detailed: dict, 
                            formated_dir: str
                           ):
        """
        Helper method to retrieve a cached cropped object or crop it from the source image.
        Returns the cropped CV image and its unique key, or (None, key) on failure.
        """
        obj_key = f"{image_name}_obj_{obj_id}"
        
        # 1. Check cache
        if obj_key in self.cropped_obj_cv:
            return self.cropped_obj_cv[obj_key], obj_key

        # 2. Validate input and metadata availability
        if image_name not in objects_data_map_detailed:
            print(f"Warning: Image name '{image_name}' not found in object predictions. Object key: {obj_key}.")
            return None, obj_key
        if obj_id not in objects_data_map_detailed[image_name]:
            print(f"Warning: Object ID {obj_id} for image '{image_name}' not found in predictions. Object key: {obj_key}.")
            return None, obj_key

        obj_pred_data = objects_data_map_detailed[image_name][obj_id]
        obj_coords_list_of_lists = obj_pred_data.get('coords')
        
        if not obj_coords_list_of_lists:
            print(f"Warning: 'coords' not found or empty for object {obj_key}. Data: {obj_pred_data}.")
            return None, obj_key

        # 3. Validate and prepare coordinates
        try:
            if not (isinstance(obj_coords_list_of_lists, list) and 
                    len(obj_coords_list_of_lists) == 4 and 
                    all(isinstance(p, list) and len(p) == 2 for p in obj_coords_list_of_lists)):
                print(f"Warning: Coordinates for {obj_key} are not in the expected format (list of 4 lists of 2 numbers). Coords: {obj_coords_list_of_lists}.")
                return None, obj_key
            
            # crop_obb_from_image expects a flat numpy array [x0,y0, x1,y1, x2,y2, x3,y3]
            obj_coords_np_flat = np.array(obj_coords_list_of_lists, dtype=np.float32).flatten()
            if obj_coords_np_flat.shape != (8,):
                 print(f"Warning: Flattened coordinates for {obj_key} do not have 8 elements (expected 4x2). Actual shape: {obj_coords_np_flat.shape}. Coords: {obj_coords_list_of_lists}.")
                 return None, obj_key
        except Exception as e:
            print(f"Warning: Error processing coordinates for {obj_key}: {e}. Coords: {obj_coords_list_of_lists}.")
            return None, obj_key

        # 4. Load source image
        source_image_path = os.path.join(formated_dir, image_name)
        if not os.path.exists(source_image_path):
            print(f"Warning: Source image file '{source_image_path}' not found for object {obj_key}.")
            return None, obj_key

        img_bytes = np.fromfile(source_image_path, np.uint8)
        image_cv = cv2.imdecode(img_bytes, cv2.IMREAD_COLOR)
        if image_cv is None:
            print(f"Warning: cv2.imread failed to read source image '{source_image_path}' for object {obj_key}.")
            return None, obj_key
        
        # 5. Crop the object
        # Ensure crop_obb_from_image is available in the scope
        cropped_obj_cv = image_processing.crop_obb_from_image(image_cv, obj_coords_np_flat) 
        
        if cropped_obj_cv is not None:
            self.cropped_obj_cv[obj_key] = cropped_obj_cv
            return cropped_obj_cv, obj_key
        else:
            print(f"Warning: crop_obb_from_image returned None for {obj_key} from image '{source_image_path}'. This could be due to small/invalid dimensions or internal OpenCV errors.")
            return None, obj_key

    def load_pairs_from_paths(self,
            similarity_json_path: str,
            verified_collages_dir: str, 
            objects_preds_json_path: str, # preictions_with_embeddings.json
            formated_dir: str
        ):
        """
        Args:
            similarity_json_path (str): # each image can contain several objects, "id" is id of this object on this image
            [
                {
                    "object1": {
                        "image_name": "skv.306_керн16-20.jpg",
                        "id": 1
                    },
                    "object2": {
                        "image_name": "skv.306_керн21-25.jpg",
                        "id": 1
                    }
                }
            ]
            verified_collages_dir (str): collages has name formated as <collage_id>_collage.jpg, where "collage_id" is index of pair objects in similarity_json_path
            objects_preds_json_path (str): 
            [
                {
                    "image_name": "skv.142801_керн1-5.jpg",
                    "objects": [
                        {
                            "id": 0,
                            "coords": [
                                [
                                    500.76959228515625,
                                    565.429443359375
                                ],
                                [
                                    514.192138671875,
                                    73.75820922851562
                                ],
                                [
                                    434.09381103515625,
                                    71.57156372070312
                                ],
                                [
                                    420.6712646484375,
                                    563.2427978515625
                                ]
                            ]
                        }
                    ]
                }
            ]
            formated_dir (str): directory with formated source images, where objects_preds_json_path "coords" from
        """
        # 1. Load object predictions and structure them for easy lookup
        try:
            with open(objects_preds_json_path, 'r', encoding='utf-8') as f:
                objects_preds_list = json.load(f)
        except FileNotFoundError:
            print(f"Error: Objects predictions JSON file not found at '{objects_preds_json_path}'. Cannot load pairs.")
            return
        except json.JSONDecodeError as e:
            print(f"Error: Could not decode Objects predictions JSON file at '{objects_preds_json_path}': {e}. Cannot load pairs.")
            return

        objects_data_map_detailed = {}
        for item in objects_preds_list:
            image_name = item.get('image_name')
            objects_in_item = item.get('objects') # This should be a list

            if not image_name or objects_in_item is None: # objects_in_item can be an empty list []
                print(f"Warning: Skipping item in object predictions due to missing 'image_name' or 'objects' field: {item}")
                continue
            
            obj_by_id = {}
            if isinstance(objects_in_item, list):
                for obj in objects_in_item:
                    if not isinstance(obj, dict):
                        print(f"Warning: Object item is not a dictionary in image {image_name}: {obj}")
                        continue
                    obj_id = obj.get('id')
                    if obj_id is None:
                        print(f"Warning: Skipping object due to missing 'id' in image {image_name}: {obj}")
                        continue
                    obj_by_id[obj_id] = obj
            else:
                print(f"Warning: 'objects' field is not a list for image {image_name}: {objects_in_item}")

            objects_data_map_detailed[image_name] = obj_by_id
        
        # 2. Load similarity pairs
        try:
            with open(similarity_json_path, 'r', encoding='utf-8') as f:
                similarity_data = json.load(f)
        except FileNotFoundError:
            print(f"Error: Similarity JSON file not found at '{similarity_json_path}'. Cannot load pairs.")
            return
        except json.JSONDecodeError as e:
            print(f"Error: Could not decode Similarity JSON file at '{similarity_json_path}': {e}. Cannot load pairs.")
            return
        
        if not isinstance(similarity_data, list):
            print(f"Error: Similarity data from '{similarity_json_path}' is not a list. Cannot load pairs.")
            return

        verified_collages_files = [f for f in os.listdir(verified_collages_dir) if f.lower().endswith(image_extensions)]
        verified_collages_ids = list(map(
            lambda x: int(x[:x.find('_')]), 
            verified_collages_files
        ))

        verified_label_count = 0
        
        # from tqdm import tqdm # Optional: for progress bar visualization
        for i, pair_info in tqdm(enumerate(similarity_data), desc="Loading pairs", total=len(similarity_data)):
        # for i, pair_info in enumerate(similarity_data):
            if self.max_verified_labels and i >= self.max_verified_labels:
                continue

            if not isinstance(pair_info, dict):
                print(f"Warning: Skipping item in similarity data (index {i}) as it's not a dictionary: {pair_info}")
                continue

            obj1_spec = pair_info.get('object1')
            obj2_spec = pair_info.get('object2')

            if not obj1_spec or not obj2_spec or not isinstance(obj1_spec, dict) or not isinstance(obj2_spec, dict):
                print(f"Warning: Skipping pair (index {i}) due to missing or malformed 'object1' or 'object2' field: {pair_info}")
                continue

            obj1_img_name = obj1_spec.get('image_name')
            obj1_id = obj1_spec.get('id') # Assuming IDs are integers as per example.
            obj2_img_name = obj2_spec.get('image_name')
            obj2_id = obj2_spec.get('id')

            if None in [obj1_img_name, obj1_id, obj2_img_name, obj2_id]:
                print(f"Warning: Skipping pair (index {i}) due to missing 'image_name' or 'id' in object specs: {pair_info}")
                continue
            
            # 3. Determine label for the pair
            # collage_filename = f"{i}_collage_{obj1_id}_vs_{obj2_id}.jpg" # collage_id is the index of the pair
            # collage_path = os.path.join(verified_collages_dir, collage_filename)
            # is_similar_pair = os.path.exists(collage_path)
            
            # label_val = 1.0 if is_similar_pair else 0.0

            is_similar_pair = True if i in verified_collages_ids else False
            label_val = 1.0 if is_similar_pair else 0.0

            # # 4. Handle max_verified_labels limit
            # if is_similar_pair:
            #     if self.max_verified_labels is not None and verified_label_count >= self.max_verified_labels:
            #         # This similar pair is skipped if the limit is reached. Dissimilar pairs are not affected by this limit.
            #         continue 
            #     verified_label_count += 1
            
            # 5. Get or crop object images
            cropped_obj1_cv, obj1_key = self._get_or_crop_object(obj1_img_name, obj1_id, objects_data_map_detailed, formated_dir)
            if cropped_obj1_cv is None:
                # Warning messages are printed by _get_or_crop_object
                print(f"Info: Skipping pair (index {i}) involving '{obj1_key}' due to failure in processing object1.")
                if is_similar_pair and self.max_verified_labels is not None: # Decrement if a similar pair was counted but then skipped
                    verified_label_count -=1
                continue 
            
            cropped_obj2_cv, obj2_key = self._get_or_crop_object(obj2_img_name, obj2_id, objects_data_map_detailed, formated_dir)
            if cropped_obj2_cv is None:
                print(f"Info: Skipping pair (index {i}) involving '{obj1_key}' and '{obj2_key}' due to failure in processing object2.")
                if is_similar_pair and self.max_verified_labels is not None: # Decrement if a similar pair was counted but then skipped
                     verified_label_count -=1
                continue

            # 6. If both objects are successfully processed/retrieved, add to dataset lists
            self.pair_obj_names.append((obj1_key, obj2_key))
            self.siamese_paire_label.append(torch.tensor(label_val, device=self.device, dtype=torch.float32))
            
        print(f"Finished loading pairs. Total pairs added: {len(self.pair_obj_names)}.")
        print(f"Number of similar pairs included: {sum(1 for lbl in self.siamese_paire_label if lbl.item() == 1.0)} (max_verified_labels was: {self.max_verified_labels}).")
        print(f"Total unique objects cached in self.cropped_obj_cv: {len(self.cropped_obj_cv)}.")
    
    def make_transforms(self):
        """
        Applies a 180-degree rotation augmentation to the dataset.
        For each original pair of images, this method creates three new pairs:
        1. (rotated_image1, image2)
        2. (image1, rotated_image2)
        3. (rotated_image1, rotated_image2)
        
        The process involves:
        - Storing the 180-degree rotated versions of images in `self.cropped_obj_cv` 
          under a new name (e.g., "<original_name>_rot").
        - Appending the newly formed pairs to `self.pair_obj_names`.
        - Appending corresponding labels to `self.siamese_paire_label`. The similarity 
          label is assumed to remain the same after rotation.
        - Recalculating embeddings for any newly added rotated images using 
          `self._calc_all_objects_embeddings()`.
        - Ensuring all embeddings are on the correct device using 
          `self._place_all_embeddings_on_device()`.
        
        This method is designed to be called once. Subsequent calls will be skipped.
        """
        if self._transforms_applied:
            print("Info: _make_transforms has already been called. Skipping further augmentation.")
            return

        try:
            # Check if cv2 module and necessary functions/constants are available
            _ = cv2.rotate 
            _ = cv2.ROTATE_180
        except (NameError, AttributeError) as e: # NameError if cv2 is not imported, AttributeError if function/constant is missing
            raise ImportError(
                "OpenCV (cv2) is required for 180-degree rotation but is not properly installed/imported, "
                "or `cv2.rotate`/`cv2.ROTATE_180` is missing. "
                "Please install/update OpenCV (e.g., `pip install opencv-python`) and ensure it's imported. "
                f"Original error: {e}"
            )

        # Step 1: Identify all unique images involved in the current pairs and create their rotated versions.
        unique_img_names = set()
        # Iterate over a copy of self.pair_obj_names as good practice if list could be modified,
        # though here it's just for collecting names before modifications begin.
        for name1, name2 in list(self.pair_obj_names): 
            unique_img_names.add(name1)
            unique_img_names.add(name2)

        for img_name in unique_img_names:
            if img_name not in self.cropped_obj_cv:
                print(f"Warning: Image '{img_name}' from pair list not found in self.cropped_obj_cv. Skipping its rotation.")
                continue # Cannot rotate an image that doesn't exist in cropped_obj_cv

            rotated_img_name = f"{img_name}_rot"
            
            # Create and store the rotated image only if it hasn't been already
            if rotated_img_name not in self.cropped_obj_cv:
                original_cv_image = self.cropped_obj_cv[img_name]
                # Assuming original_cv_image is a NumPy array (standard for OpenCV images)
                rotated_cv_image = cv2.rotate(original_cv_image, cv2.ROTATE_180)
                self.cropped_obj_cv[rotated_img_name] = rotated_cv_image
        
        # Step 2: Create new pairs using original and newly created rotated images.
        # Iterate only over the pairs that existed before this augmentation started.
        num_original_pairs = len(self.pair_obj_names) 
        new_pairs_to_add = []
        new_labels_to_add = []

        for i in range(num_original_pairs):
            obj_name_1, obj_name_2 = self.pair_obj_names[i]
            original_label = self.siamese_paire_label[i]

            # Define names for potential rotated versions
            rotated_obj_name_1 = f"{obj_name_1}_rot"
            rotated_obj_name_2 = f"{obj_name_2}_rot"

            # Check if the required image components (original or rotated) exist in self.cropped_obj_cv
            # before creating a new pair. This handles cases where an original image might have been missing,
            # preventing its rotated version from being created.

            # Augmentation type 1: (rotated_img1, img2)
            if rotated_obj_name_1 in self.cropped_obj_cv and obj_name_2 in self.cropped_obj_cv:
                new_pairs_to_add.append((rotated_obj_name_1, obj_name_2))
                new_labels_to_add.append(original_label.clone()) # Clone tensor for safety
            else:
                print(f"Warning: Skipping augmented pair ({rotated_obj_name_1}, {obj_name_2}) due to missing image component(s) in self.cropped_obj_cv.")

            # Augmentation type 2: (img1, rotated_img2)
            if obj_name_1 in self.cropped_obj_cv and rotated_obj_name_2 in self.cropped_obj_cv:
                new_pairs_to_add.append((obj_name_1, rotated_obj_name_2))
                new_labels_to_add.append(original_label.clone())
            else:
                print(f"Warning: Skipping augmented pair ({obj_name_1}, {rotated_obj_name_2}) due to missing image component(s) in self.cropped_obj_cv.")

            # Augmentation type 3: (rotated_img1, rotated_img2)
            if rotated_obj_name_1 in self.cropped_obj_cv and rotated_obj_name_2 in self.cropped_obj_cv:
                new_pairs_to_add.append((rotated_obj_name_1, rotated_obj_name_2))
                new_labels_to_add.append(original_label.clone())
            else:
                print(f"Warning: Skipping augmented pair ({rotated_obj_name_1}, {rotated_obj_name_2}) due to missing image component(s) in self.cropped_obj_cv.")

        # Step 3: Append the newly generated pairs and labels to the dataset's main lists
        self.pair_obj_names.extend(new_pairs_to_add)
        self.siamese_paire_label.extend(new_labels_to_add)

        # Step 4: Calculate embeddings for any new images (i.e., rotated ones that don't have embeddings yet)
        self.calc_all_objects_embeddings()

        # Step 5: Ensure all embeddings (including new ones) are on the correct device
        self.place_all_embeddings_on_device()
        
        self._transforms_applied = True # Mark that transforms have been applied
        print(f"Info: _make_transforms applied. Added {len(new_pairs_to_add)} new pairs. Total pairs: {len(self.pair_obj_names)}.")

    def calc_all_objects_embeddings(self):
        if not self.load_embedding_model:
            return
        for key in tqdm(self.cropped_obj_cv.keys(), desc="Getting embeddings", total=len(self.cropped_obj_cv)):
            self.obj_embeddings[key] = embedding_process.get_embedding(
                self.image_processor, 
                self.embedding_model, 
                self.cropped_obj_cv[key],
                device=self.device,
                model_name_hint=self.model_name
            )
            # if key not in self.obj_embeddings:
            #     self.obj_embeddings[key] = embedding_process.get_embedding(
            #         self.image_processor, 
            #         self.embedding_model, 
            #         self.cropped_obj_cv[key],
            #         device=self.device,
            #         model_name_hint=self.model_name
            #     )
        self.num_ftrs = len(list(self.obj_embeddings.values())[0])

    def place_all_embeddings_on_device(self):
        for key in self.obj_embeddings.keys():
            self.obj_embeddings[key] = place_embedding_on_device(
                self.obj_embeddings[key], 
                device=self.device
            )
            # if self.obj_embeddings[key] is not None:
            #     if isinstance(self.obj_embeddings[key], list):
            #         self.obj_embeddings[key] = place_embedding_on_device(
            #             self.obj_embeddings[key], 
            #             device=self.device
            #         )
            #     elif isinstance(self.obj_embeddings[key], torch.Tensor):
            #         self.obj_embeddings[key] = self.obj_embeddings[key].to(self.device)

    def _set_data(self, obj_embeddings, pair_obj_names, siamese_paire_label):
        self.obj_embeddings = obj_embeddings
        self.pair_obj_names = pair_obj_names
        self.siamese_paire_label = siamese_paire_label


class SiameseNetwork(nn.Module):
    def __init__(self, num_ftrs: int, device='cuda'):
        super().__init__()
        self.device = 'cuda'
        self.pipline = nn.Sequential(
            nn.Linear(num_ftrs, 1024),
            nn.Sigmoid(),
            nn.Linear(1024, 128),
            nn.Sigmoid()
        ).to(self.device)
        
    def forward(self, embeddings):
        # embedding1 = self.pipline(embeddings[0])
        # embedding2 = self.pipline(embeddings[1])
        embedding1 = self.predict_single_embedding(embeddings[0])
        embedding2 = self.predict_single_embedding(embeddings[1])
        # Calculate the absolute difference for each pair in the batch
        abs_diff = torch.abs(embedding1 - embedding2)
        # Sum along the feature dimension (dimension 1) to get a similarity score for each pair
        similarity_scores = 1 / (1 + torch.linalg.norm(abs_diff, dim=1))
        # similarity_scores = 1 / (1 + torch.sum(abs_diff))
        return similarity_scores
    
    def predict_single_embedding(self, embedding):
        return self.pipline(embedding)
    
    def calc_two_embeddings_sim(self, emb1, emb2):
        abs_diff = torch.abs(emb1 - emb2)
        similarity_score = 1 / (1 + torch.linalg.norm(abs_diff))
        return similarity_score
