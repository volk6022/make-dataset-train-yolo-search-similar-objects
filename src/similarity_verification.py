import json
import numpy as np
from tqdm import tqdm
import cv2
import os
from numba import jit
from collections import defaultdict
import sys
from skimage.metrics import structural_similarity as ssim
from skimage.feature import local_binary_pattern

import image_processing
from settings_and_utils import delete_directory_and_contents


### verifing every single object using SIFT
@jit
def check_match_dispersion_grid(kp_coords, img_shape, grid_rows, grid_cols, min_occupied_cells_ratio):
    """
    Checks if keypoints are reasonably dispersed across an image using a grid.

    Args:
        kp_coords: A list of keypoint objects (e.g., from good_matches).
        img_shape: The shape of the image (height, width) these keypoints belong to.
        grid_rows: Number of rows in the grid.
        grid_cols: Number of columns in the grid.
        min_occupied_cells_ratio: Minimum ratio of grid cells that must contain at least one keypoint.

    Returns:
        bool: True if matches are sufficiently dispersed, False otherwise.
    """
    if not kp_coords:
        return False
    if grid_rows == 0 or grid_cols == 0:
        return False
    
    img_h, img_w = img_shape[:2]
    cell_h = img_h / grid_rows
    cell_w = img_w / grid_cols

    if cell_h == 0 or cell_w == 0:
        return False

    occupied_cells = set()
    for kp in kp_coords:
        x, y = kp  # Keypoint coordinates
        col_idx = int(x / cell_w)
        row_idx = int(y / cell_h)
        # Clamp indices to be within grid boundaries, just in case of floating point inaccuracies at edges
        col_idx = min(col_idx, grid_cols - 1)
        row_idx = min(row_idx, grid_rows - 1)
        occupied_cells.add((row_idx, col_idx))

    num_occupied_cells = len(occupied_cells)
    total_cells = grid_rows * grid_cols
    if total_cells == 0:
        return False   
    return (num_occupied_cells / total_cells) >= min_occupied_cells_ratio

def verify_pair_with_local_features(
        img1_cv,
        img2_cv,
        min_good_matches,
        ratio_thresh,
        method,
        knnMatch_k,
        flann_index_kdtree,
        flann_index_kdtree_trees,
        search_params_checks,
        saving_good_matches_path,
        check_dispersion, 
        dispersion_grid_rows,
        dispersion_grid_cols,
        dispersion_min_occupied_cells_ratio
    ):
    if img1_cv is None or img2_cv is None:
        return False, 0, "Input image(s) is None"

    gray1 = cv2.cvtColor(img1_cv, cv2.COLOR_BGR2GRAY) if len(img1_cv.shape) == 3 else img1_cv
    gray2 = cv2.cvtColor(img2_cv, cv2.COLOR_BGR2GRAY) if len(img2_cv.shape) == 3 else img2_cv

    if method not in ['SIFT', 'ORB']:
        print(f"Warning: Method '{method}' not recognized. Defaulting to SIFT.")
        method = 'SIFT'

    kp1, des1, kp2, des2 = None, None, None, None
    matches = []

    try:
        if method == 'SIFT':
            sift = cv2.SIFT_create()
            kp1, des1 = sift.detectAndCompute(gray1, None)
            kp2, des2 = sift.detectAndCompute(gray2, None)
            if des1 is None or des2 is None or len(des1) < knnMatch_k or len(des2) < knnMatch_k:
                return False, 0, "Not enough SIFT descriptors"
            index_params = dict(algorithm=flann_index_kdtree, trees=flann_index_kdtree_trees)
            search_params = dict(checks=search_params_checks)
            flann = cv2.FlannBasedMatcher(index_params, search_params)
            matches = flann.knnMatch(des1, des2, k=knnMatch_k)
        elif method == 'ORB':
            orb = cv2.ORB_create(nfeatures=1000)
            kp1, des1 = orb.detectAndCompute(gray1, None)
            kp2, des2 = orb.detectAndCompute(gray2, None)
            if des1 is None or des2 is None or len(des1) < knnMatch_k or len(des2) < knnMatch_k:
                return False, 0, "Not enough ORB descriptors"
            bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=False)
            matches = bf.knnMatch(des1, des2, k=knnMatch_k)
    except cv2.error as e:
        return False, 0, f"OpenCV error during feature detection/matching: {e}"
    except Exception as e:
        return False, 0, f"Unexpected error during feature detection/matching: {e}"

    good_matches = []
    if matches:
        for m_n_tuple in matches:
            if len(m_n_tuple) >= 2:
                m, n = m_n_tuple[:2]
                if m.distance < ratio_thresh * n.distance:
                    good_matches.append(m)
            elif len(m_n_tuple) == 1 and knnMatch_k == 1:
                good_matches.append(m_n_tuple[0])

    num_good_matches = len(good_matches)
    if num_good_matches < min_good_matches:
        return False, num_good_matches, f"Not enough good matches ({num_good_matches} < {min_good_matches})"

    if check_dispersion and num_good_matches > 0:
        # Get keypoint coordinates from img1 (queryImage)
        good_kp1_coords = [kp1[m.queryIdx].pt for m in good_matches]
        is_dispersed_img1 = check_match_dispersion_grid(
            good_kp1_coords,
            gray1.shape,
            grid_rows=dispersion_grid_rows,
            grid_cols=dispersion_grid_cols,
            min_occupied_cells_ratio=dispersion_min_occupied_cells_ratio
        )
        # Get keypoint coordinates from img2 (trainImage)
        good_kp2_coords = [kp2[m.trainIdx].pt for m in good_matches]
        is_dispersed_img2 = check_match_dispersion_grid(
            good_kp2_coords,
            gray2.shape,
            grid_rows=dispersion_grid_rows,
            grid_cols=dispersion_grid_cols,
            min_occupied_cells_ratio=dispersion_min_occupied_cells_ratio
        )
        if not is_dispersed_img1 and not is_dispersed_img2:
            return False, num_good_matches, "Matches not well-dispersed in both images"
        elif not is_dispersed_img1:
            return False, num_good_matches, "Matches not well-dispersed in image 1"
        elif not is_dispersed_img2:
            return False, num_good_matches, "Matches not well-dispersed in image 2"
        # If both are dispersed, continue

    is_verified_pair = True 
    status_message_base = f"Verified: {num_good_matches} good matches"
    if check_dispersion:
        if num_good_matches > 0:
            status_message = f"{status_message_base}, well-dispersed in both images."
        else:
            status_message = f"{status_message_base} (dispersion not checked: no matches)."
    else:
        status_message = f"{status_message_base} (dispersion check disabled)."


    if saving_good_matches_path is not None:
        try:
            src_pts = np.float32([kp1[m.queryIdx].pt for m in good_matches]).reshape(-1, 1, 2)
            dst_pts = np.float32([kp2[m.trainIdx].pt for m in good_matches]).reshape(-1, 1, 2)
            M, mask = None, None
            if len(src_pts) >= 4:
                M, mask = cv2.findHomography(src_pts, dst_pts, cv2.RANSAC, 5.0)
            matchesMask = [0]*len(good_matches)
            if M is not None and mask is not None:
                matchesMask = mask.ravel().tolist()
            else:
                print(f"Warning: Homography computation failed for {saving_good_matches_path}. Drawing all good matches.")

            img_to_draw_on = gray2.copy()
            if M is not None: # Check M is not None before using it
                h_gray1, w_gray1 = gray1.shape[:2] # Ensure we take shape of grayscale image
                pts_bounds = np.float32([[0, 0], [0, h_gray1 - 1], [w_gray1 - 1, h_gray1 - 1], [w_gray1 - 1, 0]]).reshape(-1, 1, 2)
                dst_bounds = cv2.perspectiveTransform(pts_bounds, M)
                img_to_draw_on = cv2.polylines(img_to_draw_on, [np.int32(dst_bounds)], True, (0,0,255), 3, cv2.LINE_AA)
            draw_params = dict(
                matchColor=(0, 255, 0),
                singlePointColor=None,
                matchesMask=matchesMask,
                flags=cv2.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS
            )
            if M is None or not any(matchesMask):
                draw_params['matchesMask'] = None
            img3 = cv2.drawMatches(
                gray1, kp1,
                img_to_draw_on, kp2,
                good_matches, None,
                **draw_params
            )
            cv2.imwrite(saving_good_matches_path, img3)
        except Exception as e:
            print(f'\nError during saving good match image\nmatch name: {saving_good_matches_path}\nerror: {e}')
            status_message += f" (Error saving image: {e})"

    return is_verified_pair, num_good_matches, status_message


# --- Lazy-loaded model caches (avoid reloading on every pair) ---
_xfeat_instance = None
_lightglue_cache = {}


def _get_xfeat():
    global _xfeat_instance
    if _xfeat_instance is None:
        import torch
        _xfeat_instance = torch.hub.load(
            'verlab/accelerated_features', 'XFeat', pretrained=True, trust_repo=True
        )
    return _xfeat_instance


def _get_lightglue(extractor_name, max_keypoints, device):
    global _lightglue_cache
    key = (extractor_name, max_keypoints, device)
    if key not in _lightglue_cache:
        from lightglue import LightGlue, SuperPoint, DISK, ALIKED
        extractor_map = {'superpoint': SuperPoint, 'disk': DISK, 'aliked': ALIKED}
        ExtractorClass = extractor_map.get(extractor_name.lower(), SuperPoint)
        extractor = ExtractorClass(max_num_keypoints=max_keypoints).eval().to(device)
        matcher = LightGlue(features=extractor_name.lower()).eval().to(device)
        _lightglue_cache[key] = (extractor, matcher)
    return _lightglue_cache[key]


def verify_pair_with_xfeat(
        img1_cv,
        img2_cv,
        min_good_matches,
        top_k,
        saving_good_matches_path,
        check_dispersion,
        dispersion_grid_rows,
        dispersion_grid_cols,
        dispersion_min_occupied_cells_ratio
    ):
    """Verify a pair using XFeat neural feature matching (torch.hub: verlab/accelerated_features)."""
    if img1_cv is None or img2_cv is None:
        return False, 0, "Input image(s) is None"
    try:
        xfeat = _get_xfeat()
        img1_rgb = cv2.cvtColor(img1_cv, cv2.COLOR_BGR2RGB) if len(img1_cv.shape) == 3 else img1_cv
        img2_rgb = cv2.cvtColor(img2_cv, cv2.COLOR_BGR2RGB) if len(img2_cv.shape) == 3 else img2_cv

        mkpts0, mkpts1 = xfeat.match_xfeat(img1_rgb, img2_rgb, top_k=top_k)
        num_matches = len(mkpts0)

        if num_matches < min_good_matches:
            return False, num_matches, f"Not enough XFeat matches ({num_matches} < {min_good_matches})"

        if check_dispersion and num_matches > 0:
            pts1 = [(float(p[0]), float(p[1])) for p in mkpts0]
            pts2 = [(float(p[0]), float(p[1])) for p in mkpts1]
            is_disp1 = check_match_dispersion_grid(
                pts1, img1_cv.shape, dispersion_grid_rows, dispersion_grid_cols,
                dispersion_min_occupied_cells_ratio
            )
            is_disp2 = check_match_dispersion_grid(
                pts2, img2_cv.shape, dispersion_grid_rows, dispersion_grid_cols,
                dispersion_min_occupied_cells_ratio
            )
            if not is_disp1 and not is_disp2:
                return False, num_matches, "XFeat matches not dispersed in both images"
            elif not is_disp1:
                return False, num_matches, "XFeat matches not dispersed in image 1"
            elif not is_disp2:
                return False, num_matches, "XFeat matches not dispersed in image 2"

        if saving_good_matches_path is not None:
            try:
                kp1 = [cv2.KeyPoint(float(p[0]), float(p[1]), 1) for p in mkpts0]
                kp2 = [cv2.KeyPoint(float(p[0]), float(p[1]), 1) for p in mkpts1]
                cv_matches = [cv2.DMatch(i, i, 0) for i in range(num_matches)]
                vis = cv2.drawMatches(
                    img1_cv, kp1, img2_cv, kp2, cv_matches, None,
                    flags=cv2.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS
                )
                cv2.imwrite(saving_good_matches_path, vis)
            except Exception as e:
                pass  # non-fatal

        return True, num_matches, f"XFeat verified: {num_matches} matches"

    except Exception as e:
        return False, 0, f"XFeat error: {e}"


def verify_pair_with_lightglue(
        img1_cv,
        img2_cv,
        min_good_matches,
        extractor_name,
        max_keypoints,
        confidence_threshold,
        device,
        saving_good_matches_path,
        check_dispersion,
        dispersion_grid_rows,
        dispersion_grid_cols,
        dispersion_min_occupied_cells_ratio
    ):
    """Verify a pair using LightGlue + SuperPoint/DISK/ALIKED neural feature matching."""
    if img1_cv is None or img2_cv is None:
        return False, 0, "Input image(s) is None"
    try:
        import torch
        extractor, matcher = _get_lightglue(extractor_name, max_keypoints, device)

        def cv2_to_tensor(img):
            rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB) if len(img.shape) == 3 else img
            t = torch.from_numpy(rgb).float() / 255.0
            return t.permute(2, 0, 1).unsqueeze(0).to(device)  # [1, 3, H, W]

        img0_t = cv2_to_tensor(img1_cv)
        img1_t = cv2_to_tensor(img2_cv)

        with torch.no_grad():
            feats0 = extractor.extract(img0_t)
            feats1 = extractor.extract(img1_t)
            result = matcher({'image0': feats0, 'image1': feats1})

        # Remove batch dimension
        from lightglue.utils import rbd
        feats0, feats1, result = [rbd(x) for x in [feats0, feats1, result]]

        matches = result['matches']  # (K, 2)
        scores = result.get('scores', None)

        # Filter by confidence threshold if scores available
        if scores is not None and confidence_threshold > 0:
            mask = scores >= confidence_threshold
            matches = matches[mask]

        if len(matches) == 0:
            return False, 0, "No LightGlue matches after confidence filtering"

        mkpts0 = feats0['keypoints'][matches[:, 0]].cpu().numpy()
        mkpts1 = feats1['keypoints'][matches[:, 1]].cpu().numpy()
        num_matches = len(mkpts0)

        if num_matches < min_good_matches:
            return False, num_matches, f"Not enough LightGlue matches ({num_matches} < {min_good_matches})"

        if check_dispersion and num_matches > 0:
            pts1 = [(float(p[0]), float(p[1])) for p in mkpts0]
            pts2 = [(float(p[0]), float(p[1])) for p in mkpts1]
            is_disp1 = check_match_dispersion_grid(
                pts1, img1_cv.shape, dispersion_grid_rows, dispersion_grid_cols,
                dispersion_min_occupied_cells_ratio
            )
            is_disp2 = check_match_dispersion_grid(
                pts2, img2_cv.shape, dispersion_grid_rows, dispersion_grid_cols,
                dispersion_min_occupied_cells_ratio
            )
            if not is_disp1 and not is_disp2:
                return False, num_matches, "LightGlue matches not dispersed in both images"
            elif not is_disp1:
                return False, num_matches, "LightGlue matches not dispersed in image 1"
            elif not is_disp2:
                return False, num_matches, "LightGlue matches not dispersed in image 2"

        if saving_good_matches_path is not None:
            try:
                kp1 = [cv2.KeyPoint(float(p[0]), float(p[1]), 1) for p in mkpts0]
                kp2 = [cv2.KeyPoint(float(p[0]), float(p[1]), 1) for p in mkpts1]
                cv_matches = [cv2.DMatch(i, i, 0) for i in range(num_matches)]
                vis = cv2.drawMatches(
                    img1_cv, kp1, img2_cv, kp2, cv_matches, None,
                    flags=cv2.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS
                )
                cv2.imwrite(saving_good_matches_path, vis)
            except Exception as e:
                pass  # non-fatal

        return True, num_matches, f"LightGlue ({extractor_name}) verified: {num_matches} matches"

    except Exception as e:
        return False, 0, f"LightGlue error: {e}"


# Helper function to log-transform Hu moments (typically called before calculating distance)
def _log_transform_hu_moments_vector(hu_moments_vec):
    """
    Log-transforms a vector of Hu moments.
    The transformation is h_i' = sign(h_i) * log10(|h_i|).
    Handles cases where h_i is zero or very small to avoid -inf.

    Args:
        hu_moments_vec (np.ndarray): A 1D numpy array of 7 Hu moments.

    Returns:
        np.ndarray: A 1D numpy array of 7 log-transformed Hu moments.
    """
    transformed = np.zeros_like(hu_moments_vec, dtype=float)
    for i in range(len(hu_moments_vec)):
        val = hu_moments_vec[i]
        abs_val = abs(val)
        
        # Threshold for "effectively zero" to prevent log10(0) -> -inf
        # log10(1e-20) is -20. Values smaller than this would go further to -inf.
        if abs_val < 1e-20: 
            # If a raw moment is effectively zero, its log-transformed value can be set to 0.
            # The difference calculation |0 - log_transformed_other_moment| will handle it.
            # Or, can be set to a large negative like -20.0 if that fits the metric better.
            # Setting to 0.0 for simplicity here.
            transformed[i] = 0.0 
        else:
            transformed_val = np.copysign(1.0, val) * np.log10(abs_val)
            # Cap extremely small values that might still result in -inf or very large negatives
            transformed[i] = max(transformed_val, -20.0) # Cap at -20 (log10(1e-20))
    return transformed

# Assumed helper function (based on its usage in the main function)
def _get_merged_contour_from_selection(selected_contours, image_shape, debug_masks_list=None):
    """
    Creates a single "super-contour" by drawing selected contours onto a mask
    and then re-extracting the external contour of the resulting shape.
    """
    mask_for_visualization = np.full(image_shape, 64, dtype=np.uint8) # Default gray mask

    if not selected_contours:
        if debug_masks_list is not None:
             debug_masks_list.append(cv2.cvtColor(mask_for_visualization, cv2.COLOR_GRAY2BGR))
        return None

    mask = np.zeros(image_shape, dtype=np.uint8)
    cv2.drawContours(mask, selected_contours, -1, (255), thickness=cv2.FILLED)
    
    mask_for_visualization = mask.copy() # Use the actual mask for visualization

    if debug_masks_list is not None:
        mask_vis_bgr = cv2.cvtColor(mask_for_visualization, cv2.COLOR_GRAY2BGR)
        debug_masks_list.append(mask_vis_bgr)

    merged_contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    if not merged_contours:
        return None

    merged_contours = sorted(merged_contours, key=cv2.contourArea, reverse=True)
    final_merged_contour = merged_contours[0]
    
    # Optionally, draw the found merged contour back onto the debug mask
    # if debug_masks_list is not None and len(debug_masks_list) > 0:
    #    cv2.drawContours(debug_masks_list[-1], [final_merged_contour], -1, (0,255,255), 1) # Yellow

    return final_merged_contour

def verify_pair_with_contours( # Function name kept same as per "make same function"
    img1_cv,
    img2_cv,
    binary_threshold_type,
    fixed_threshold_value,
    adaptive_block_size,
    adaptive_C,
    # contour_match_shapes_method, # Removed: Hu moments comparison is now specific
    contour_similarity_threshold, # Lower is better for Hu moment distance
    min_contour_area_ratio,
    save_contour_debug_image_path,
    top_n_contours_to_compare=1 # Number of largest contours to consider for messages/logic
):
    """
    Verifies if two images are similar based on Hu moment comparison of their "super-contours".
    Selects significant contours from img1, creates a "super-contour" from their union
    (by drawing on a mask and re-extracting). Does the same for img2.
    Compares these two "super-contours" using their log-transformed Hu moments and L1 distance.
    Returns: (is_verified, similarity_score, status_message)
             similarity_score is float('inf') if error or no match possible.
    """
    if img1_cv is None or img2_cv is None:
        return False, float('inf'), "Input image(s) is None"

    gray1 = cv2.cvtColor(img1_cv, cv2.COLOR_BGR2GRAY) if len(img1_cv.shape) == 3 else img1_cv.copy()
    gray2 = cv2.cvtColor(img2_cv, cv2.COLOR_BGR2GRAY) if len(img2_cv.shape) == 3 else img2_cv.copy()

    try:
        if binary_threshold_type == "otsu":
            _, thresh1 = cv2.threshold(gray1, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
            _, thresh2 = cv2.threshold(gray2, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
        elif binary_threshold_type == "adaptive":
            thresh1 = cv2.adaptiveThreshold(gray1, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                                            cv2.THRESH_BINARY_INV, adaptive_block_size, adaptive_C)
            thresh2 = cv2.adaptiveThreshold(gray2, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                                            cv2.THRESH_BINARY_INV, adaptive_block_size, adaptive_C)
        elif binary_threshold_type == "fixed":
            _, thresh1 = cv2.threshold(gray1, fixed_threshold_value, 255, cv2.THRESH_BINARY_INV)
            _, thresh2 = cv2.threshold(gray2, fixed_threshold_value, 255, cv2.THRESH_BINARY_INV)
        else:
            return False, float('inf'), f"Unknown binary_threshold_type: {binary_threshold_type}"
    except cv2.error as e:
        return False, float('inf'), f"OpenCV error during thresholding: {e}"

    contours1, _ = cv2.findContours(thresh1, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    contours2, _ = cv2.findContours(thresh2, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    min_area1 = max(1, img1_cv.shape[0] * img1_cv.shape[1] * min_contour_area_ratio)
    min_area2 = max(1, img2_cv.shape[0] * img2_cv.shape[1] * min_contour_area_ratio)

    significant_contours1 = [c for c in contours1 if cv2.contourArea(c) > min_area1]
    significant_contours2 = [c for c in contours2 if cv2.contourArea(c) > min_area2]

    if not significant_contours1 or not significant_contours2:
        return False, float('inf'), (f"Not enough significant contours to proceed (img1: {len(significant_contours1)}, "
                                     f"img2: {len(significant_contours2)})")

    sorted_sig_contours1 = sorted(significant_contours1, key=cv2.contourArea, reverse=True)
    sorted_sig_contours2 = sorted(significant_contours2, key=cv2.contourArea, reverse=True)

    user_requested_n_contours = top_n_contours_to_compare
    actual_n_to_select = max(1, user_requested_n_contours) # Used for messages

    # Current logic merges all significant contours, not just top_n_contours_to_compare
    selected_for_merge1 = sorted_sig_contours1[:] 
    selected_for_merge2 = sorted_sig_contours2[:]
    
    if not selected_for_merge1 or not selected_for_merge2: # Should be caught by significant_contours check
         return False, float('inf'), (f"Selected list for merging is empty (img1: {len(selected_for_merge1)}, "
                                     f"img2: {len(selected_for_merge2)}).")

    debug_masks_for_visualization = [] if save_contour_debug_image_path else None

    final_contour1 = _get_merged_contour_from_selection(selected_for_merge1, gray1.shape,
                                                        debug_masks_for_visualization)
    final_contour2 = _get_merged_contour_from_selection(selected_for_merge2, gray2.shape,
                                                        debug_masks_for_visualization)

    if final_contour1 is None or final_contour2 is None:
        msg = "Failed to create merged super-contour from selected contours "
        if final_contour1 is None: msg += "(img1: no contour from mask) "
        if final_contour2 is None: msg += "(img2: no contour from mask)"
        return False, float('inf'), msg

    # --- Hu Moment Comparison ---
    moments1 = cv2.moments(final_contour1)
    hu_moments1_raw = cv2.HuMoments(moments1).flatten() # flatten from (7,1) to (7,)
    
    moments2 = cv2.moments(final_contour2)
    hu_moments2_raw = cv2.HuMoments(moments2).flatten()

    # Log-transform the Hu moments for more robust comparison
    hu1_transformed = _log_transform_hu_moments_vector(hu_moments1_raw)
    hu2_transformed = _log_transform_hu_moments_vector(hu_moments2_raw)

    # Calculate similarity score using L1 norm (Manhattan distance) on transformed moments
    # This is analogous to cv2.matchShapes method CONTOURS_MATCH_I1
    similarity_score = np.sum(np.abs(hu1_transformed - hu2_transformed))
    # --- End Hu Moment Comparison ---

    is_verified = similarity_score < contour_similarity_threshold

    num_contours_used_for_merge_img1 = len(selected_for_merge1)
    num_contours_used_for_merge_img2 = len(selected_for_merge2)
    
    details_msg = ""
    # The message reflects if the number used differs from what top_n might imply
    # or if top_n was not 1 (which is often the default for "single main object")
    if user_requested_n_contours != 1 or \
       num_contours_used_for_merge_img1 < actual_n_to_select or \
       num_contours_used_for_merge_img2 < actual_n_to_select or \
       (actual_n_to_select == 1 and (num_contours_used_for_merge_img1 > 1 or num_contours_used_for_merge_img2 > 1) ):
        # Added case for when actual_n_to_select is 1 but more were merged (due to current merging strategy)
        details_msg = (f"(Used top {num_contours_used_for_merge_img1} for img1 merge & "
                       f"top {num_contours_used_for_merge_img2} for img2 merge; "
                       f"N_req={user_requested_n_contours}, N_attempt_select_logic={actual_n_to_select}) ")
                       # N_attempt_select_logic refers to how many would be chosen if top_N was strictly followed for selection

    status_message = (f"{details_msg}Hu Moments L1 dist: {similarity_score:.4f} "
                      f"(Thr: <{contour_similarity_threshold})")
    status_message += " - Verified" if is_verified else " - No match"
        
    # Kept the original condition for saving debug image, assuming comment was misleading
    if save_contour_debug_image_path and is_verified: 
        try:
            img1_disp = img1_cv.copy()
            img2_disp = img2_cv.copy()
            
            if selected_for_merge1:
                cv2.drawContours(img1_disp, selected_for_merge1, -1, (0, 255, 0), 2)
            if selected_for_merge2:
                cv2.drawContours(img2_disp, selected_for_merge2, -1, (0, 0, 255), 2)
            
            thresh1_color = cv2.cvtColor(thresh1, cv2.COLOR_GRAY2BGR)
            thresh2_color = cv2.cvtColor(thresh2, cv2.COLOR_GRAY2BGR)

            mask1_vis_final = (debug_masks_for_visualization[0]
                               if debug_masks_for_visualization and len(debug_masks_for_visualization) > 0
                               else cv2.cvtColor(np.full(gray1.shape, 64, dtype=np.uint8), cv2.COLOR_GRAY2BGR))
            mask2_vis_final = (debug_masks_for_visualization[1]
                               if debug_masks_for_visualization and len(debug_masks_for_visualization) > 1
                               else cv2.cvtColor(np.full(gray2.shape, 64, dtype=np.uint8), cv2.COLOR_GRAY2BGR))

            rows_data = [
                (img1_disp, img2_disp),
                (thresh1_color, thresh2_color),
                (mask1_vis_final, mask2_vis_final)
            ]
            
            processed_rows = []
            max_total_width = 0

            for img_left, img_right in rows_data:
                if len(img_left.shape) == 2: img_left = cv2.cvtColor(img_left, cv2.COLOR_GRAY2BGR)
                if len(img_right.shape) == 2: img_right = cv2.cvtColor(img_right, cv2.COLOR_GRAY2BGR)

                h_left, h_right = img_left.shape[0], img_right.shape[0]
                max_h_pair = max(h_left, h_right)

                if h_left < max_h_pair:
                    img_left = cv2.copyMakeBorder(img_left, (max_h_pair - h_left) // 2, max_h_pair - h_left - ((max_h_pair - h_left) // 2),
                                                  0, 0, cv2.BORDER_CONSTANT, value=(128,128,128))
                if h_right < max_h_pair:
                    img_right = cv2.copyMakeBorder(img_right, (max_h_pair - h_right) // 2, max_h_pair - h_right - ((max_h_pair - h_right) // 2),
                                                   0, 0, cv2.BORDER_CONSTANT, value=(128,128,128))
                
                stitched_row = np.hstack((img_left, img_right))
                processed_rows.append(stitched_row)
                if stitched_row.shape[1] > max_total_width:
                    max_total_width = stitched_row.shape[1]

            final_rows_to_vstack = []
            for row_img in processed_rows:
                if row_img.shape[1] < max_total_width:
                    row_img = cv2.copyMakeBorder(row_img, 0, 0, 0, max_total_width - row_img.shape[1],
                                                 cv2.BORDER_CONSTANT, value=(0,0,0))
                final_rows_to_vstack.append(row_img)
            
            debug_img = np.vstack(final_rows_to_vstack)
            
            text_color = (0,0,255) if not is_verified else (0,255,0)
            cv2.putText(debug_img, f"Hu L1: {similarity_score:.4f} {'VERIFIED' if is_verified else 'NO MATCH'}", 
                        (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, text_color, 2)
            
            # Updated details_line_txt to be more informative about N logic
            n_actual_sel_info = f"N_req:{user_requested_n_contours}, N_sel_logic:{actual_n_to_select}"
            details_line_txt = (f"N_merged: img1={num_contours_used_for_merge_img1}, img2={num_contours_used_for_merge_img2} "
                                f"({n_actual_sel_info}) Thr:{contour_similarity_threshold:.2f}")
            cv2.putText(debug_img, details_line_txt, 
                        (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.5, text_color, 1)
            cv2.putText(debug_img, "Row1:Orig+Selected, R2:Thresh, R3:MasksForMerge", (10, debug_img.shape[0] - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (200,200,200), 1)

            cv2.imwrite(save_contour_debug_image_path, debug_img)
        except Exception as e:
            err_msg = f"Error saving contour debug image {save_contour_debug_image_path}: {type(e).__name__} - {e}"
            import traceback
            err_msg += f"\nTraceback: {traceback.format_exc()}"
            print(err_msg, file=sys.stderr) # Ensure output to stderr for errors
            status_message += f" (Error saving debug: {type(e).__name__})"

    return is_verified, similarity_score, status_message


def verify_pair_with_image_features(
    img1_cv,
    img2_cv,
    method,
    similarity_threshold,
    # Edge Detection (Canny)
    canny_threshold1=100,
    canny_threshold2=150,
    # Corner Detection (Shi-Tomasi)
    gftt_max_corners=200,
    gftt_quality_level=0.01,
    gftt_min_distance=10,
    # Blob Detection (SimpleBlobDetector)
    blob_min_area=50,
    blob_max_area=10000,
    blob_min_circularity=0.1,
    # Texture Analysis (LBP)
    lbp_radius=1,
    lbp_points=8, # P = 8 * R
    # General
    saving_visualization_path=None,
    resize_for_comparison=True
):
    """
    Compares two images based on selected image features (Edge, Corner, Blob, Texture).

    Args:
        img1_cv (np.ndarray): First input image (OpenCV format, BGR or Grayscale).
        img2_cv (np.ndarray): Second input image (OpenCV format, BGR or Grayscale).
        method (str): The comparison method. Options: 'EDGE', 'CORNER', 'BLOB', 'TEXTURE'.
        similarity_threshold (float): Threshold for considering images similar.
                                      - For 'EDGE', 'CORNER', 'BLOB' (SSIM-based): Value between 0 and 1. Higher is more similar.
                                      - For 'TEXTURE' (LBP Histogram Correlation): Value between 0 and 1. Higher is more similar.
        canny_threshold1 (int): First threshold for Canny edge detector.
        canny_threshold2 (int): Second threshold for Canny edge detector.
        gftt_max_corners (int): Max corners for Shi-Tomasi.
        gftt_quality_level (float): Quality level for Shi-Tomasi.
        gftt_min_distance (int): Min distance between corners for Shi-Tomasi.
        blob_min_area (float): Minimum area for blob detection.
        blob_max_area (float): Maximum area for blob detection.
        blob_min_circularity (float): Minimum circularity for blob detection.
        lbp_radius (int): Radius for LBP.
        lbp_points (int): Number of points for LBP (P in LBP_P,R notation).
        saving_visualization_path (str, optional): Path to save a visualization image. Defaults to None.
        resize_for_comparison (bool): If True, resizes the second image to match the first for methods
                                      that require same-sized inputs (EDGE, CORNER, BLOB with SSIM).

    Returns:
        tuple: (is_verified_pair, score, status_message)
               - is_verified_pair (bool): True if images are considered similar based on the threshold.
               - score (float): The calculated similarity score.
               - status_message (str): A message describing the outcome.
    """
    if img1_cv is None or img2_cv is None:
        return False, 0.0, "Input image(s) is None"

    gray1 = cv2.cvtColor(img1_cv, cv2.COLOR_BGR2GRAY) if len(img1_cv.shape) == 3 else img1_cv.copy()
    gray2 = cv2.cvtColor(img2_cv, cv2.COLOR_BGR2GRAY) if len(img2_cv.shape) == 3 else img2_cv.copy()

    # Ensure images are of type uint8 for many OpenCV functions
    if gray1.dtype != np.uint8: gray1 = cv2.normalize(gray1, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
    if gray2.dtype != np.uint8: gray2 = cv2.normalize(gray2, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)


    valid_methods = ['EDGE', 'CORNER', 'BLOB', 'TEXTURE']
    if method.upper() not in valid_methods:
        return False, 0.0, f"Method '{method}' not recognized. Valid methods: {valid_methods}"
    method = method.upper()

    score = 0.0
    is_verified_pair = False
    feature_map1, feature_map2 = None, None # For visualization

    try:
        h1, w1 = gray1.shape
        h2, w2 = gray2.shape
        gray2_resized = gray2

        if resize_for_comparison and (h1 != h2 or w1 != w2) and method in ['EDGE', 'CORNER', 'BLOB']:
            # print(f"Resizing image 2 ({w2}x{h2}) to match image 1 ({w1}x{h1}) for {method} comparison.")
            gray2_resized = cv2.resize(gray2, (w1, h1), interpolation=cv2.INTER_AREA)
        elif method in ['EDGE', 'CORNER', 'BLOB'] and (h1 != h2 or w1 != w2) and not resize_for_comparison:
             return False, 0.0, f"{method} comparison requires same-sized images or resize_for_comparison=True. Got {w1}x{h1} and {w2}x{h2}."


        if method == 'EDGE':
            edges1 = cv2.Canny(gray1, canny_threshold1, canny_threshold2)
            edges2 = cv2.Canny(gray2_resized, canny_threshold1, canny_threshold2)
            
            # Ensure inputs are single-channel and float32
            if edges1.ndim > 2: edges1 = cv2.cvtColor(edges1, cv2.COLOR_BGR2GRAY)
            if edges2.ndim > 2: edges2 = cv2.cvtColor(edges2, cv2.COLOR_BGR2GRAY)
            img1_float = edges1.astype(np.float32)
            img2_float = edges2.astype(np.float32)
            
            # Template matching requires template to be smaller or equal.
            # If they are same size (which they are after preprocessing), result is 1x1 array.
            if img1_float.shape[0] > img2_float.shape[0] or img1_float.shape[1] > img2_float.shape[1]:
                # swap if img1 is larger than img2 (template must be smaller)
                result = cv2.matchTemplate(img1_float, img2_float, cv2.TM_CCORR_NORMED)
            else:
                result = cv2.matchTemplate(img2_float, img1_float, cv2.TM_CCORR_NORMED)
            score = float(np.average(result))

            # score = ssim(edges1, edges2, data_range=edges1.max() - edges1.min())
            is_verified_pair = score >= similarity_threshold
            feature_map1, feature_map2 = edges1, edges2

        elif method == 'CORNER':
            # Create blank images to draw corners on for SSIM comparison
            corners_img1 = np.zeros_like(gray1)
            corners_img2 = np.zeros_like(gray2_resized)

            corners1 = cv2.goodFeaturesToTrack(gray1, gftt_max_corners, gftt_quality_level, gftt_min_distance)
            corners2 = cv2.goodFeaturesToTrack(gray2_resized, gftt_max_corners, gftt_quality_level, gftt_min_distance)

            if corners1 is not None:
                for c in corners1:
                    x, y = c.ravel().astype(int)
                    cv2.circle(corners_img1, (x, y), 3, 255, -1) # Draw white circles
            if corners2 is not None:
                for c in corners2:
                    x, y = c.ravel().astype(int)
                    cv2.circle(corners_img2, (x, y), 3, 255, -1)

            # score = ssim(corners_img1, corners_img2, data_range=corners_img1.max() - corners_img1.min())
            # Ensure inputs are single-channel and float32
            if corners_img1.ndim > 2: corners_img1 = cv2.cvtColor(corners_img1, cv2.COLOR_BGR2GRAY)
            if corners_img2.ndim > 2: corners_img2 = cv2.cvtColor(corners_img2, cv2.COLOR_BGR2GRAY)
            img1_float = corners_img1.astype(np.float32)
            img2_float = corners_img2.astype(np.float32)
            
            # Template matching requires template to be smaller or equal.
            # If they are same size (which they are after preprocessing), result is 1x1 array.
            if img1_float.shape[0] > img2_float.shape[0] or img1_float.shape[1] > img2_float.shape[1]:
                # swap if img1 is larger than img2 (template must be smaller)
                result = cv2.matchTemplate(img1_float, img2_float, cv2.TM_CCORR_NORMED)
            else:
                result = cv2.matchTemplate(img2_float, img1_float, cv2.TM_CCORR_NORMED)
            score = float(np.average(result))

            is_verified_pair = score >= similarity_threshold
            feature_map1, feature_map2 = corners_img1, corners_img2

        elif method == 'BLOB':
            params = cv2.SimpleBlobDetector_Params()
            params.minThreshold = 10
            params.maxThreshold = 200
            params.filterByArea = True
            params.minArea = blob_min_area
            params.maxArea = blob_max_area
            params.filterByCircularity = True
            params.minCircularity = blob_min_circularity
            params.filterByConvexity = False # Can be enabled
            # params.minConvexity = 0.87
            params.filterByInertia = False # Can be enabled
            # params.minInertiaRatio = 0.01

            detector = cv2.SimpleBlobDetector_create(params)

            keypoints1 = detector.detect(gray1)
            keypoints2 = detector.detect(gray2_resized) # Use resized gray2

            # Create blank images to draw blobs on for SSIM comparison
            blobs_img1 = np.zeros_like(gray1)
            blobs_img2 = np.zeros_like(gray2_resized) # Use shape of resized gray2

            blobs_img1 = cv2.drawKeypoints(gray1, keypoints1, np.array([]), (0,0,255), # Draw on gray to see context
                                          cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)
            blobs_img2 = cv2.drawKeypoints(gray2_resized, keypoints2, np.array([]), (0,0,255),
                                          cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)
            
            # For SSIM, better to compare binary masks or feature presence
            # Let's draw on black canvas for a cleaner SSIM
            blobs_mask1 = np.zeros_like(gray1)
            blobs_mask2 = np.zeros_like(gray2_resized)
            for kp in keypoints1:
                cv2.circle(blobs_mask1, (int(kp.pt[0]), int(kp.pt[1])), int(kp.size / 2), 255, -1)
            for kp in keypoints2:
                cv2.circle(blobs_mask2, (int(kp.pt[0]), int(kp.pt[1])), int(kp.size / 2), 255, -1)

            score = ssim(blobs_mask1, blobs_mask2, data_range=blobs_mask1.max() - blobs_mask1.min())
            is_verified_pair = score >= similarity_threshold
            feature_map1, feature_map2 = blobs_img1, blobs_img2 # For visualization, show drawn on original

        elif method == 'TEXTURE':
            # LBP parameters
            n_points = lbp_points
            radius = lbp_radius
            METHOD_LBP = 'default' # 'default', 'ror', 'uniform', 'nri_uniform', 'var'

            lbp1 = local_binary_pattern(gray1, n_points, radius, METHOD_LBP)
            lbp2 = local_binary_pattern(gray2, n_points, radius, METHOD_LBP) # Original gray2, LBP handles different sizes

            # Calculate histograms
            # For 'uniform' LBP, the number of bins is n_points + 2.
            # For other methods, it's 2**n_points.
            if METHOD_LBP == 'uniform':
                bins_lbp = n_points + 2
            else:
                bins_lbp = 2**n_points
            
            hist1, _ = np.histogram(lbp1.ravel(), bins=np.arange(0, bins_lbp + 1), range=(0, bins_lbp))
            hist2, _ = np.histogram(lbp2.ravel(), bins=np.arange(0, bins_lbp + 1), range=(0, bins_lbp))

            # Normalize histograms
            hist1 = hist1.astype("float")
            hist1 /= (hist1.sum() + 1e-6) # Add epsilon to avoid division by zero
            hist2 = hist2.astype("float")
            hist2 /= (hist2.sum() + 1e-6)

            # Compare histograms using correlation
            score = cv2.compareHist(hist1, hist2, cv2.HISTCMP_CORREL)
            is_verified_pair = score >= similarity_threshold
            # For visualization, show the LBP images themselves
            # Normalize LBP images to 0-255 for display
            feature_map1 = cv2.normalize(lbp1, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
            feature_map2 = cv2.normalize(lbp2, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)

    except cv2.error as e:
        return False, 0.0, f"OpenCV error during {method} processing: {e}"
    except Exception as e:
        return False, 0.0, f"Unexpected error during {method} processing: {e}"

    status_message = f"{method} comparison: Score = {score:.4f}. "
    status_message += "Verified." if is_verified_pair else "Not verified."
    status_message += f" (Threshold: {similarity_threshold})"

    if is_verified_pair and saving_visualization_path and feature_map1 is not None and feature_map2 is not None:
        try:
            # Ensure feature maps are 3-channel for stacking if originals are color
            vis_fm1 = cv2.cvtColor(feature_map1, cv2.COLOR_GRAY2BGR) if len(feature_map1.shape) == 2 else feature_map1
            vis_fm2 = cv2.cvtColor(feature_map2, cv2.COLOR_GRAY2BGR) if len(feature_map2.shape) == 2 else feature_map2
            
            # Original images (ensure they are BGR for stacking)
            orig_img1_bgr = img1_cv if len(img1_cv.shape) == 3 else cv2.cvtColor(gray1, cv2.COLOR_GRAY2BGR)
            orig_img2_bgr = img2_cv if len(img2_cv.shape) == 3 else cv2.cvtColor(gray2, cv2.COLOR_GRAY2BGR)

            # Resize feature maps to match original display if necessary
            # (especially if gray2_resized was used for feature_map2 computation)
            if vis_fm1.shape[:2] != orig_img1_bgr.shape[:2]:
                 vis_fm1 = cv2.resize(vis_fm1, (orig_img1_bgr.shape[1], orig_img1_bgr.shape[0]))
            if vis_fm2.shape[:2] != orig_img2_bgr.shape[:2]:
                 vis_fm2 = cv2.resize(vis_fm2, (orig_img2_bgr.shape[1], orig_img2_bgr.shape[0]))


            # Create a combined visualization
            # Top row: original images, Bottom row: feature maps/visualizations
            # Ensure all parts have the same height for hstack by resizing if needed
            # For simplicity, let's just stack img1 with feature_map1 and img2 with feature_map2, then stack horizontally
            
            # Make sure all images for stacking are the same height
            # If gray2 was resized for comparison, feature_map2 might be smaller than orig_img2_bgr
            # Let's resize feature maps to match their corresponding original images for a nice side-by-side
            h_orig1, w_orig1 = orig_img1_bgr.shape[:2]
            h_orig2, w_orig2 = orig_img2_bgr.shape[:2]

            vis_fm1_resized = cv2.resize(vis_fm1, (w_orig1, h_orig1))
            vis_fm2_resized = cv2.resize(vis_fm2, (w_orig2, h_orig2))

            # Stack original above its feature map
            col1 = np.vstack((orig_img1_bgr, vis_fm1_resized))
            col2 = np.vstack((orig_img2_bgr, vis_fm2_resized))

            # Ensure both columns have same height before hstack
            target_h = max(col1.shape[0], col2.shape[0])
            if col1.shape[0] != target_h:
                col1 = cv2.resize(col1, (col1.shape[1], target_h))
            if col2.shape[0] != target_h:
                col2 = cv2.resize(col2, (col2.shape[1], target_h))

            final_visualization = np.hstack((col1, col2))
            
            title_text = f"{method} | Score: {score:.3f} | Thresh: {similarity_threshold} | Verified: {is_verified_pair}"
            # Add text to the image
            cv2.putText(final_visualization, title_text, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 
                        1, (0, 255, 0) if is_verified_pair else (0,0,255), 2, cv2.LINE_AA)

            cv2.imwrite(saving_visualization_path, final_visualization)
        except Exception as e:
            print(f'\nError during saving visualization image\nPath: {saving_visualization_path}\nError: {e}')
            status_message += f" (Error saving visualization: {e})"

    return is_verified_pair, score, status_message


def verify_similarities(
        predictions_with_embeddings_path,
        sorted_pairs_path,
        formated_images_folder,
        verification_method, # 'local_features' or 'contours' or 'image_features'
        verify_180_rotation=True,
        max_pairs=None, # cap pairs before verification; None = no limit
        clean_debug_dir=True,
        # Local feature parameters
        local_features_min_good_matches=10, 
        local_features_ratio_thresh=0.8, 
        local_features_method='SIFT', 
        local_features_knnMatch_k=2,
        local_features_flann_index_kdtree=1,
        local_features_flann_index_kdtree_trees=20,
        local_features_search_params_checks=100,
        local_features_saving_good_matches_dir = None,
        local_features_check_dispersion=True,
        local_features_dispersion_grid_rows=10,
        local_features_dispersion_grid_cols=2,
        local_features_dispersion_min_occupied_cells_ratio=0.7,
        # Contour verification parameters
        contour_binary_threshold_type="otsu", # "otsu", "adaptive", "fixed"
        contour_fixed_threshold_value=127,
        contour_adaptive_block_size=11, # Should be odd
        contour_adaptive_C=2,
        contour_match_shapes_method_str="I1", # "I1", "I2", "I3"
        contour_similarity_threshold=0.7, # Lower is better for matchShapes
        contour_min_contour_area_ratio=0.01, # Min contour area relative to image area
        saving_contour_debug_dir=None,
        contour_top_n_contours_to_compare=1,
        # image features parameters
        image_features_debug_dir = None,
        image_features_method = 'EDGE', # 'EDGE', 'CORNER', 'BLOB', 'TEXTURE'
        image_features_similarity_threshold = 0.5,
        # XFeat parameters (used when local_features_method == 'XFEAT')
        xfeat_top_k = 4096,
        # LightGlue parameters (used when local_features_method == 'LIGHTGLUE')
        lightglue_extractor = 'superpoint',  # 'superpoint', 'disk', 'aliked'
        lightglue_max_keypoints = 2048,
        lightglue_confidence_threshold = 0.5,
        lightglue_device = 'cpu'
    ):
    print(f"Starting verification using method: {verification_method}...")
#     print(f"""Common parameters:
# predictions_with_embeddings_path: {predictions_with_embeddings_path}
# sorted_pairs_path: {sorted_pairs_path}
# formated_images_folder: {formated_images_folder}""")

    if verification_method == 'local_features':
        if clean_debug_dir:
            delete_directory_and_contents(local_features_saving_good_matches_dir)
        if local_features_saving_good_matches_dir:
            os.makedirs(local_features_saving_good_matches_dir, exist_ok=True)
        if local_features_method not in ['SIFT', 'ORB', 'XFEAT', 'LIGHTGLUE']:
            print(f"Warning: Unknown local_features_method '{local_features_method}', defaulting to SIFT.")
            local_features_method = 'SIFT'
    elif verification_method == 'contours':
#         print(f"""Contour verification parameters:
# binary_threshold_type: {contour_binary_threshold_type}
# fixed_threshold_value: {contour_fixed_threshold_value} (if type is 'fixed')
# adaptive_block_size: {contour_adaptive_block_size} (if type is 'adaptive')
# adaptive_C: {contour_adaptive_C} (if type is 'adaptive')
# match_shapes_method: {contour_match_shapes_method_str}
# similarity_threshold: {contour_similarity_threshold} (lower is better)
# min_contour_area_ratio: {contour_min_contour_area_ratio}
# saving_contour_debug_dir: {saving_contour_debug_dir}
# contour_top_n_contours_to_compare: {contour_top_n_contours_to_compare}""")
        if contour_adaptive_block_size % 2 == 0:
            print(f"Warning: contour_adaptive_block_size ({contour_adaptive_block_size}) should be odd. Adjusting to {contour_adaptive_block_size + 1}.")
            contour_adaptive_block_size +=1
        if clean_debug_dir:
            delete_directory_and_contents(saving_contour_debug_dir)
        if saving_contour_debug_dir:
            os.makedirs(saving_contour_debug_dir, exist_ok=True)
        
        contour_match_shapes_method_map = {
            "I1": cv2.CONTOURS_MATCH_I1,
            "I2": cv2.CONTOURS_MATCH_I2,
            "I3": cv2.CONTOURS_MATCH_I3
        }
        actual_contour_match_method = contour_match_shapes_method_map.get(contour_match_shapes_method_str.upper())
        if actual_contour_match_method is None:
            print(f"Warning: Invalid contour_match_shapes_method_str '{contour_match_shapes_method_str}'. Defaulting to I1.")
            actual_contour_match_method = cv2.CONTOURS_MATCH_I1
    elif verification_method == 'image_features':
        if clean_debug_dir:
            delete_directory_and_contents(image_features_debug_dir)
        if image_features_debug_dir:
            os.makedirs(image_features_debug_dir, exist_ok=True)
        if image_features_method not in ['EDGE', 'CORNER', 'BLOB', 'TEXTURE']:
            image_features_method = 'EDGE'
            print(f"invalid image_features_method, must be in ['EDGE', 'CORNER', 'BLOB', 'TEXTURE'], using : {image_features_method}")
    else:
        print(f"Error: Unknown verification_method '{verification_method}'. Exiting.")
        return []

    with open(predictions_with_embeddings_path, 'r', encoding='utf-8') as f:
        predictions_with_embeddings = json.load(f)
    with open(sorted_pairs_path, 'r', encoding='utf-8') as f:
        sorted_pairs = json.load(f)

    if max_pairs is not None and len(sorted_pairs) > max_pairs:
        print(f"Limiting verification to top {max_pairs} of {len(sorted_pairs)} candidate pairs.")
        sorted_pairs = sorted_pairs[:max_pairs]

    potential_matches_count = len(sorted_pairs)
    num_digits_for_sequence = len(str(potential_matches_count)) if potential_matches_count > 0 else 1

    verified_pairs = []

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
    for image_name, obj_ids_in_image in pbar_preprocessing:
        img_path = os.path.join(formated_images_folder, image_name)
        cv_image_full = None
        try:
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
            try:
                cropped_obj_cv = image_processing.crop_obb_from_image(cv_image_full, coords_np)
                cropped_object_cache[object_key] = cropped_obj_cv
            except Exception as e:
                print(f"Error cropping {object_key}: {e}")
                cropped_object_cache[object_key] = None
    
    num_total_unique_objects = len(unique_objects_to_crop)
    num_successfully_cropped = sum(1 for obj_cv in cropped_object_cache.values() if obj_cv is not None)
    num_failed_crop_or_load = num_total_unique_objects - num_successfully_cropped
    
    print(f"Preprocessing finished. Attempted: {num_total_unique_objects} unique objects.")
    print(f"Successfully cropped: {num_successfully_cropped}. Failed/Skipped: {num_failed_crop_or_load}.")

    for i in tqdm(range(len(sorted_pairs)), desc=f"Verifying pairs ({verification_method})"):
        pair = sorted_pairs[i]
        obj1_info = pair["object1"]
        obj2_info = pair["object2"]

        obj1_key = (obj1_info["image_name"], obj1_info["id"])
        obj2_key = (obj2_info["image_name"], obj2_info["id"])

        cropped_obj1_cv = cropped_object_cache.get(obj1_key)
        cropped_obj2_cv = cropped_object_cache.get(obj2_key)

        if cropped_obj1_cv is None or cropped_obj2_cv is None:
            continue
        
        sequence_str = str(i).zfill(num_digits_for_sequence)
        clean_img1_name = "".join(c if c.isalnum() else "_" for c in obj1_info["image_name"])
        clean_obj1_id = "".join(c if c.isalnum() else "_" for c in str(obj1_info["id"]))
        clean_img2_name = "".join(c if c.isalnum() else "_" for c in obj2_info["image_name"])
        clean_obj2_id = "".join(c if c.isalnum() else "_" for c in str(obj2_info["id"]))

        is_verified_final = False
        final_verification_details = {}

        # --- Verification Attempt 1: Original Orientation ---
        if verification_method == 'local_features':
            saving_path_attempt1 = None
            if local_features_saving_good_matches_dir:
                output_filename_attempt1 = f"{sequence_str}_match_{clean_img1_name}_{clean_obj1_id}_vs_{clean_img2_name}_{clean_obj2_id}_orig.jpg"
                saving_path_attempt1 = os.path.join(local_features_saving_good_matches_dir, output_filename_attempt1)

            if local_features_method in ['SIFT', 'ORB']:
                is_verified_attempt1, num_matches_attempt1, _ = verify_pair_with_local_features(
                    img1_cv=cropped_obj1_cv,
                    img2_cv=cropped_obj2_cv,
                    min_good_matches=local_features_min_good_matches,
                    ratio_thresh=local_features_ratio_thresh,
                    method=local_features_method,
                    knnMatch_k=local_features_knnMatch_k,
                    flann_index_kdtree=local_features_flann_index_kdtree,
                    flann_index_kdtree_trees=local_features_flann_index_kdtree_trees,
                    search_params_checks=local_features_search_params_checks,
                    saving_good_matches_path=saving_path_attempt1,
                    check_dispersion=local_features_check_dispersion,
                    dispersion_grid_rows=local_features_dispersion_grid_rows,
                    dispersion_grid_cols=local_features_dispersion_grid_cols,
                    dispersion_min_occupied_cells_ratio=local_features_dispersion_min_occupied_cells_ratio
                )
            elif local_features_method == 'XFEAT':
                is_verified_attempt1, num_matches_attempt1, _ = verify_pair_with_xfeat(
                    img1_cv=cropped_obj1_cv,
                    img2_cv=cropped_obj2_cv,
                    min_good_matches=local_features_min_good_matches,
                    top_k=xfeat_top_k,
                    saving_good_matches_path=saving_path_attempt1,
                    check_dispersion=local_features_check_dispersion,
                    dispersion_grid_rows=local_features_dispersion_grid_rows,
                    dispersion_grid_cols=local_features_dispersion_grid_cols,
                    dispersion_min_occupied_cells_ratio=local_features_dispersion_min_occupied_cells_ratio
                )
            elif local_features_method == 'LIGHTGLUE':
                is_verified_attempt1, num_matches_attempt1, _ = verify_pair_with_lightglue(
                    img1_cv=cropped_obj1_cv,
                    img2_cv=cropped_obj2_cv,
                    min_good_matches=local_features_min_good_matches,
                    extractor_name=lightglue_extractor,
                    max_keypoints=lightglue_max_keypoints,
                    confidence_threshold=lightglue_confidence_threshold,
                    device=lightglue_device,
                    saving_good_matches_path=saving_path_attempt1,
                    check_dispersion=local_features_check_dispersion,
                    dispersion_grid_rows=local_features_dispersion_grid_rows,
                    dispersion_grid_cols=local_features_dispersion_grid_cols,
                    dispersion_min_occupied_cells_ratio=local_features_dispersion_min_occupied_cells_ratio
                )
            else:
                is_verified_attempt1, num_matches_attempt1 = False, 0

            if is_verified_attempt1:
                is_verified_final = True
                final_verification_details["local_feature_matches"] = num_matches_attempt1
                final_verification_details["rotation_verified"] = False
        
        elif verification_method == 'contours':
            saving_path_attempt1 = None
            if saving_contour_debug_dir:
                output_filename_attempt1 = f"{sequence_str}_contourmatch_{clean_img1_name}_{clean_obj1_id}_vs_{clean_img2_name}_{clean_obj2_id}_orig.jpg"
                saving_path_attempt1 = os.path.join(saving_contour_debug_dir, output_filename_attempt1)

            is_verified_attempt1, score_attempt1, _ = verify_pair_with_contours(
                img1_cv=cropped_obj1_cv,
                img2_cv=cropped_obj2_cv,
                binary_threshold_type=contour_binary_threshold_type,
                fixed_threshold_value=contour_fixed_threshold_value,
                adaptive_block_size=contour_adaptive_block_size,
                adaptive_C=contour_adaptive_C,
                # contour_match_shapes_method=actual_contour_match_method,
                contour_similarity_threshold=contour_similarity_threshold,
                min_contour_area_ratio=contour_min_contour_area_ratio,
                save_contour_debug_image_path=saving_path_attempt1,
                top_n_contours_to_compare=contour_top_n_contours_to_compare
            )
            if is_verified_attempt1:
                is_verified_final = True
                final_verification_details["contour_similarity_score"] = score_attempt1
                final_verification_details["rotation_verified"] = False
            
        elif verification_method == 'image_features':
            saving_path_attempt1 = None
            if image_features_debug_dir:
                output_filename_attempt1 = f"{sequence_str}_image_features_match_{clean_img1_name}_{clean_obj1_id}_vs_{clean_img2_name}_{clean_obj2_id}_orig.jpg"
                saving_path_attempt1 = os.path.join(image_features_debug_dir, output_filename_attempt1)

            is_verified_attempt1, score_attempt1, _ = verify_pair_with_image_features(
                img1_cv=cropped_obj1_cv,
                img2_cv=cropped_obj2_cv,
                method=image_features_method,
                similarity_threshold=image_features_similarity_threshold,
                saving_visualization_path=saving_path_attempt1
            )
            if is_verified_attempt1:
                is_verified_final = True
                final_verification_details["image_features_similarity_score"] = score_attempt1
                final_verification_details["rotation_verified"] = False

        # --- Verification Attempt 2: Rotated img2 (if enabled and not already verified) ---
        if verify_180_rotation and not is_verified_final:
            # Ensure cropped_obj2_cv is valid before rotating
            if cropped_obj2_cv is not None:
                rotated_cropped_obj2_cv = cv2.rotate(cropped_obj2_cv, cv2.ROTATE_180)
                
                if verification_method == 'local_features':
                    saving_path_attempt2 = None
                    if local_features_saving_good_matches_dir:
                        output_filename_attempt2 = f"{sequence_str}_match_{clean_img1_name}_{clean_obj1_id}_vs_{clean_img2_name}_{clean_obj2_id}_rot180.jpg"
                        saving_path_attempt2 = os.path.join(local_features_saving_good_matches_dir, output_filename_attempt2)

                    if local_features_method in ['SIFT', 'ORB']:
                        is_verified_attempt2, num_matches_attempt2, _ = verify_pair_with_local_features(
                            img1_cv=cropped_obj1_cv,
                            img2_cv=rotated_cropped_obj2_cv,
                            min_good_matches=local_features_min_good_matches,
                            ratio_thresh=local_features_ratio_thresh,
                            method=local_features_method,
                            knnMatch_k=local_features_knnMatch_k,
                            flann_index_kdtree=local_features_flann_index_kdtree,
                            flann_index_kdtree_trees=local_features_flann_index_kdtree_trees,
                            search_params_checks=local_features_search_params_checks,
                            saving_good_matches_path=saving_path_attempt2,
                            check_dispersion=local_features_check_dispersion,
                            dispersion_grid_rows=local_features_dispersion_grid_rows,
                            dispersion_grid_cols=local_features_dispersion_grid_cols,
                            dispersion_min_occupied_cells_ratio=local_features_dispersion_min_occupied_cells_ratio
                        )
                    elif local_features_method == 'XFEAT':
                        is_verified_attempt2, num_matches_attempt2, _ = verify_pair_with_xfeat(
                            img1_cv=cropped_obj1_cv,
                            img2_cv=rotated_cropped_obj2_cv,
                            min_good_matches=local_features_min_good_matches,
                            top_k=xfeat_top_k,
                            saving_good_matches_path=saving_path_attempt2,
                            check_dispersion=local_features_check_dispersion,
                            dispersion_grid_rows=local_features_dispersion_grid_rows,
                            dispersion_grid_cols=local_features_dispersion_grid_cols,
                            dispersion_min_occupied_cells_ratio=local_features_dispersion_min_occupied_cells_ratio
                        )
                    elif local_features_method == 'LIGHTGLUE':
                        is_verified_attempt2, num_matches_attempt2, _ = verify_pair_with_lightglue(
                            img1_cv=cropped_obj1_cv,
                            img2_cv=rotated_cropped_obj2_cv,
                            min_good_matches=local_features_min_good_matches,
                            extractor_name=lightglue_extractor,
                            max_keypoints=lightglue_max_keypoints,
                            confidence_threshold=lightglue_confidence_threshold,
                            device=lightglue_device,
                            saving_good_matches_path=saving_path_attempt2,
                            check_dispersion=local_features_check_dispersion,
                            dispersion_grid_rows=local_features_dispersion_grid_rows,
                            dispersion_grid_cols=local_features_dispersion_grid_cols,
                            dispersion_min_occupied_cells_ratio=local_features_dispersion_min_occupied_cells_ratio
                        )
                    else:
                        is_verified_attempt2, num_matches_attempt2 = False, 0

                    if is_verified_attempt2:
                        is_verified_final = True
                        final_verification_details["local_feature_matches"] = num_matches_attempt2
                        final_verification_details["rotation_verified"] = True
                
                elif verification_method == 'contours':
                    saving_path_attempt2 = None
                    if saving_contour_debug_dir:
                        output_filename_attempt2 = f"{sequence_str}_contourmatch_{clean_img1_name}_{clean_obj1_id}_vs_{clean_img2_name}_{clean_obj2_id}_rot180.jpg"
                        saving_path_attempt2 = os.path.join(saving_contour_debug_dir, output_filename_attempt2)

                    is_verified_attempt2, score_attempt2, _ = verify_pair_with_contours(
                        img1_cv=cropped_obj1_cv,
                        img2_cv=rotated_cropped_obj2_cv,
                        binary_threshold_type=contour_binary_threshold_type,
                        fixed_threshold_value=contour_fixed_threshold_value,
                        adaptive_block_size=contour_adaptive_block_size,
                        adaptive_C=contour_adaptive_C,
                        # contour_match_shapes_method=actual_contour_match_method,
                        contour_similarity_threshold=contour_similarity_threshold,
                        min_contour_area_ratio=contour_min_contour_area_ratio,
                        save_contour_debug_image_path=saving_path_attempt2,
                        top_n_contours_to_compare=contour_top_n_contours_to_compare
                    )
                    if is_verified_attempt2:
                        is_verified_final = True
                        final_verification_details["contour_similarity_score"] = score_attempt2
                        final_verification_details["rotation_verified"] = True

                elif verification_method == 'image_features':
                    saving_path_attempt2 = None
                    if image_features_debug_dir:
                        output_filename_attempt2 = f"{sequence_str}_image_features_match_{clean_img1_name}_{clean_obj1_id}_vs_{clean_img2_name}_{clean_obj2_id}_rot180.jpg"
                        saving_path_attempt2 = os.path.join(image_features_debug_dir, output_filename_attempt2)
                    is_verified_attempt2, score_attempt2, _ = verify_pair_with_image_features(
                        img1_cv=cropped_obj1_cv,
                        img2_cv=rotated_cropped_obj2_cv,
                        method=image_features_method,
                        similarity_threshold=image_features_similarity_threshold,
                        saving_visualization_path=saving_path_attempt2
                    )
                    if is_verified_attempt2:
                        is_verified_final = True
                        final_verification_details["image_features_similarity_score"] = score_attempt2
                        final_verification_details["rotation_verified"] = True
        if is_verified_final:
            final_verification_details["pair_id"] = i
            pair.update(final_verification_details)
            verified_pairs.append(pair)
    
    print(f"Found {len(sorted_pairs)} initial pairs, {len(verified_pairs)} after {verification_method} verification.")
    return verified_pairs


