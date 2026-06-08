import json
import numpy as np
from itertools import combinations
from tqdm import tqdm
import torch

# Dispatch dictionary for comparison functions
# All functions should take two 1D Torch tensors (embeddings) and return a scalar similarity score.
# Embeddings are L2 normalized by place_all_embeddings_on_device before being passed to these functions.
_comparison_functions = {
    "cosine": lambda e1, e2: torch.dot(e1, e2),  # Cosine similarity for normalized vectors
    "euclidean": lambda e1, e2: 1 / (1 + torch.linalg.norm(e1 - e2)), # Inverse Euclidean distance based similarity
                                                                    # For normalized vectors, this is related to angular distance.
                                                                    # dist = 0 -> sim = 1; dist -> 2 (max for normalized) -> sim = 1/3
    "manhattan": lambda e1, e2: 1 / (1 + torch.sum(torch.abs(e1 - e2))), # Inverse Manhattan distance based similarity
}


### comparing embeddings
def load_and_preprocess_data(json_filepath: str):
    """
    Loads data from a JSON file and preprocesses it.
    Converts embedding lists to 1D NumPy arrays.
    Filters out objects with invalid or zero-norm embeddings.
    Flattens the structure to a list of all objects.
    """
    try:
        with open(json_filepath, 'r', encoding='utf-8') as f:
            data = json.load(f)
    except FileNotFoundError:
        print(f"Error: JSON file not found at {json_filepath}")
        return []
    except json.JSONDecodeError:
        print(f"Error: Could not decode JSON from {json_filepath}")
        return []

    all_objects = []
    for image_data in data:
        image_name = image_data.get("image_name", "unknown_image")
        for obj in image_data.get("objects", []):
            obj_id = obj.get("id", -1)
            embedding_list = obj.get("embedding")

            if embedding_list is None:
                print(f"Warning: Object ID {obj_id} in {image_name} has no embedding. Skipping.")
                continue
            
            try:
                if not isinstance(embedding_list, list) or \
                   not all(isinstance(x, (int, float)) for x in embedding_list):
                    print(f"Warning: Embedding for Object ID {obj_id} in {image_name} is not a valid list of numbers. Skipping. Found: {embedding_list}")
                    continue
                
                embedding_np = np.array(embedding_list, dtype=float)
                if embedding_np.ndim != 1 or embedding_np.size == 0:
                    print(f"Warning: Embedding for Object ID {obj_id} in {image_name} is not a valid 1D vector or is empty. Skipping. Shape: {embedding_np.shape}")
                    continue
                
                if np.linalg.norm(embedding_np) < 1e-9: # Check for zero or near-zero norm
                    print(f"Warning: Embedding for Object ID {obj_id} in {image_name} has zero or near-zero norm. Skipping.")
                    continue

            except Exception as e:
                print(f"Error converting/validating embedding for Object ID {obj_id} in {image_name}: {e}. Skipping.")
                continue

            all_objects.append({
                "image_name": image_name,
                "id": obj_id,
                "embedding": embedding_np, # Store as 1D NumPy array
                "original_object_ref": obj
            })
    
    if not all_objects:
        print("No valid objects with embeddings found in the JSON data.")
    elif len(all_objects) == 1:
        print("Only one object with embedding found. Cannot perform comparisons.")

    return all_objects

def place_all_embeddings_on_device(objects: list, device: torch.device):
    """
    Moves embeddings to the specified torch.device and normalizes them (L2 normalization).
    Assumes embeddings are non-zero norm based on pre-filtering in load_and_preprocess_data.
    """
    for i in range(len(objects)):
        emb_np = objects[i]["embedding"] # This is a 1D numpy array
        emb = torch.tensor(emb_np, device=device, dtype=torch.float32) # Converts to 1D tensor
        
        norm = torch.linalg.norm(emb)
        if norm > 1e-9: # Safeguard against near-zero norms
            emb = emb / norm
        else:
            # This case should ideally not be hit if load_and_preprocess_data filters zero-norm vectors.
            # If it is, treat as a zero vector to prevent NaN/Inf.
            print(f"Warning: Object ID {objects[i].get('id', 'N/A')} in image {objects[i].get('image_name', 'N/A')} has a near-zero norm embedding ({norm.item()}) on device. Using zero vector.")
            emb = torch.zeros_like(emb) 

        objects[i]["embedding"] = emb


def compare_object_pair_device(obj1_data, obj2_data, sim_tresh, method="cosine"):
    """
    Compares a single pair of objects on the specified device using the given similarity method
    and returns similarity information if above threshold.

    Args:
        obj1_data (dict): Data for the first object, with 'embedding' as a 1D normalized Torch tensor.
        obj2_data (dict): Data for the second object, with 'embedding' as a 1D normalized Torch tensor.
        sim_tresh (float): Similarity threshold. Pairs with similarity below this are ignored.
                           The interpretation of this threshold depends on the method.
        method (str): The similarity calculation method. Supported: "cosine", "euclidean", "manhattan".

    Returns:
        dict or None: Similarity information if above threshold, else None.
                      The dict includes an additional 'method' key indicating the comparison method used.
    """
    if obj1_data["image_name"] == obj2_data["image_name"]:
        return None

    emb1 = obj1_data["embedding"]
    emb2 = obj2_data["embedding"]

    if method not in _comparison_functions:
        # This check is also present in find_most_similar_objects_diff_images, but good for direct use too
        raise ValueError(f"Unknown comparison method: {method}. Supported methods are: {list(_comparison_functions.keys())}")

    comparison_func = _comparison_functions[method]
    sim_score = comparison_func(emb1, emb2).item()

    if sim_score > sim_tresh:
        return {
            "object1": {
                "image_name": obj1_data["image_name"],
                "id": obj1_data["id"],
                # "embedding": emb1.cpu().numpy().tolist()
            },
            "object2": {
                "image_name": obj2_data["image_name"],
                "id": obj2_data["id"],
                # "embedding": emb2.cpu().numpy().tolist()
            },
            "similarity": sim_score,
            "method": method # Include the method in the result
        }
    return None

def find_most_similar_objects_diff_images(objects: list, sim_tresh=0.7, device="cpu", method="cosine"):
    """
    Compares unique pairs of objects from DIFFERENT images using the specified device and similarity method.
    Embeddings are L2-normalized before comparison.

    Args:
        objects (list): List of object data dictionaries, preprocessed by load_and_preprocess_data.
        sim_tresh (float): Similarity threshold. Pairs with similarity below this are ignored.
                           Default is 0.7, suitable for 'cosine' similarity (range [0,1] for typical embeddings).
                           For 'euclidean' or 'manhattan' methods (which use 1/(1+distance)),
                           similarity is also in [0,1] (approx [1/3, 1] for normalized vectors with Euclidean).
                           A sim_tresh of 0.7 for these would correspond to a small distance.
                           E.g., for Euclidean on unit sphere, d < 0.42 => sim > 0.7.
        device (str): Device to use for calculations (e.g., "cpu", "cuda", "cuda:0", "mps").
        method (str): The similarity calculation method. Supported: "cosine", "euclidean", "manhattan".
                      Default is "cosine".

    Returns:
        list: A sorted list of dictionaries, where each dictionary represents a similar pair:
              [{'object1': {'image_name': ..., 'id': ...},
                'object2': {'image_name': ..., 'id': ...},
                'similarity': ...,
                'method': ...}, ...]
              Sorted by similarity in descending order.
    """
    if len(objects) < 2:
        print("Not enough objects to compare.")
        return []

    if method not in _comparison_functions:
        print(f"Error: Unknown comparison method '{method}'. Supported methods are: {list(_comparison_functions.keys())}")
        return []

    object_pairs = list(combinations(objects, 2))
    print(f"Total number of object pairs to compare: {len(object_pairs)}")

    # Determine device
    resolved_device_obj: torch.device
    try:
        resolved_device_obj = torch.device(device)
        # Perform a minimal check if the device is available and functional
        if resolved_device_obj.type == 'cuda' and not torch.cuda.is_available():
            raise RuntimeError("CUDA specified but not available")
        if resolved_device_obj.type == 'mps' and (not hasattr(torch.backends, 'mps') or not torch.backends.mps.is_available()):
            raise RuntimeError("MPS specified but not available")
        
        # For non-CPU devices, try a small tensor operation to ensure it's functional
        if resolved_device_obj.type != 'cpu':
             torch.tensor([1.0], device=resolved_device_obj) # Test operation
    except RuntimeError as e:
        print(f"Warning: Device '{device}' not available or could not be initialized ({e}). Falling back to CPU.")
        resolved_device_obj = torch.device("cpu")
    except Exception as e: # Catch other potential errors for device initialization
        print(f"Warning: An unexpected error occurred while setting device '{device}': {e}. Falling back to CPU.")
        resolved_device_obj = torch.device("cpu")
        
    print(f"Using device: {resolved_device_obj}")
    
    place_all_embeddings_on_device(objects, resolved_device_obj)

    object_pairs_similarities = []
    for i, (obj1_data, obj2_data) in enumerate(tqdm(object_pairs, desc=f"Comparing object pairs using {method} method")):
        result = compare_object_pair_device(obj1_data, obj2_data, sim_tresh, method=method)
        if result:
            result["pair_id"] = i
            # if result["similarity"] > 0.7:
            #     continue
            object_pairs_similarities.append(result)
        # if len(object_pairs_similarities) >= 5000:
        #     break

    # Sort by similarity in descending order
    sorted_pairs = sorted(object_pairs_similarities, key=lambda x: x["similarity"], reverse=True)

    # if not sorted_pairs:
    #     print(f"No similar pairs found between different images using method '{method}' with threshold {sim_tresh}.")
    # else:
    #     print(f"Found {len(sorted_pairs)} similar pairs using method '{method}' with threshold {sim_tresh}.")

    # return sorted_pairs[:5000]
    return sorted_pairs