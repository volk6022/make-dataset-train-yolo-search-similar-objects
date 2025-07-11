import cv2
import numpy as np
import pandas as pd
from skimage import metrics, feature, filters
from skimage.transform import resize as skimage_resize
from skimage.color import rgb2gray
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import warnings
from matplotlib.offsetbox import OffsetImage, AnnotationBbox # New import
import matplotlib.image as mpimg
import matplotlib.gridspec as gridspec
import os


def create_similarity_report_with_images(
        csv_filepath, 
        images_folder, 
        output_filename="similarity_report.png",
        ref_img_size=(400, 50, 3)
        ):
    """
    Creates a composite image showing a heatmap of similarity metrics and
    the corresponding reference and compared images below it.

    Args:
        csv_filepath (str): Path to the CSV file with similarity results.
        images_folder (str): Path to the folder containing the images.
        output_filename (str): Filename for the output composite image.
    """
    try:
        df = pd.read_csv(csv_filepath)
    except FileNotFoundError:
        print(f"Error: CSV file not found at {csv_filepath}")
        return
    except Exception as e:
        print(f"Error reading CSV file: {e}")
        return

    if df.empty:
        print("Error: CSV file is empty.")
        return

    # --- 1. Data Preparation ---
    reference_image_name = df.iloc[0]['Image']
    compared_df = df.iloc[1:].copy()

    if compared_df.empty:
        print("Error: No comparison images found in the CSV (only reference).")
        return

    metrics_for_heatmap = [
        'MSE', 'PSNR', 'SSIM', 'NCC', 'Hist_Correl',
        'Hist_ChiSq', 'Hist_Intersect', 'Feature_Match',
        'Edge_Similarity', 'Gradient_Sim', 'Color_Hist'
    ]
    
    # Ensure only existing columns are selected from compared_df
    valid_metrics_for_heatmap = [m for m in metrics_for_heatmap if m in compared_df.columns]
    if len(valid_metrics_for_heatmap) < len(metrics_for_heatmap):
        missing = set(metrics_for_heatmap) - set(valid_metrics_for_heatmap)
        print(f"Warning: The following metric columns were not found in the CSV and will be excluded from the heatmap: {list(missing)}")
    
    if not valid_metrics_for_heatmap:
        print("Error: No valid metric columns found for the heatmap.")
        return
        
    heatmap_data = compared_df[valid_metrics_for_heatmap].copy()

    # Convert all selected metric columns to numeric.
    # 'inf', '-inf' strings (if any) will become np.inf, -np.inf floats.
    # Other non-convertible strings will become np.nan.
    for col in valid_metrics_for_heatmap:
        heatmap_data[col] = pd.to_numeric(heatmap_data[col], errors='coerce')

    # Replace np.inf and -np.inf with np.nan.
    # This ensures they are masked in the heatmap and don't skew the color scale.
    heatmap_data.replace([np.inf, -np.inf], np.nan, inplace=True)
    
    image_filenames_compared = compared_df['Image'].tolist()
    num_compared_images = len(image_filenames_compared)

    # --- 2. Plot Setup ---
    fig_height = 4 + num_compared_images * 2.5 
    fig_width = 12 
    
    fig = plt.figure(figsize=(fig_width, fig_height))
    height_ratios = [2] + [1] * num_compared_images 
    gs = gridspec.GridSpec(num_compared_images + 1, 2, height_ratios=height_ratios, hspace=0.5, wspace=0.1) # Increased hspace a bit

    # --- 3. Plot Heatmap ---
    ax_heatmap = fig.add_subplot(gs[0, :])
    
    nan_mask = heatmap_data.isnull()

    if not heatmap_data.empty:
        sns.heatmap(heatmap_data, annot=True, fmt=".2f", cmap="viridis", ax=ax_heatmap,
                    yticklabels=image_filenames_compared, xticklabels=valid_metrics_for_heatmap,
                    cbar=True, mask=nan_mask, annot_kws={"size": 8})
        
        ax_heatmap.set_title(f"Similarity Metrics Compared to '{reference_image_name}'", fontsize=14)
        
        # Configure X tick labels using plt.setp for robust ha (horizontalalignment) control
        plt.setp(ax_heatmap.get_xticklabels(), rotation=45, ha="right", rotation_mode="anchor", fontsize=8)
        # Configure Y tick labels
        plt.setp(ax_heatmap.get_yticklabels(), rotation=0, fontsize=8)
    else:
        ax_heatmap.set_title("No data for heatmap", fontsize=14)
        ax_heatmap.axis('off')


    # --- 4. Plot Images ---
    ref_img_path = os.path.join(images_folder, reference_image_name)
    try:
        ref_img = mpimg.imread(ref_img_path)
    except FileNotFoundError:
        print(f"Error: Reference image not found at {ref_img_path}")
        ref_img = np.zeros(ref_img_size, dtype=np.uint8) 
        reference_image_name_display = os.path.basename(reference_image_name) + " (Not Found)"
    except Exception as e:
        print(f"Error loading reference image {ref_img_path}: {e}")
        ref_img = np.zeros(ref_img_size, dtype=np.uint8)
        reference_image_name_display = os.path.basename(reference_image_name) + " (Load Error)"
    else:
        reference_image_name_display = os.path.basename(reference_image_name)


    for i, comp_img_name in enumerate(image_filenames_compared):
        ax_ref = fig.add_subplot(gs[1 + i, 0])
        ax_ref.imshow(ref_img)
        ax_ref.set_title(f"Reference:\n{reference_image_name_display}", fontsize=9)
        ax_ref.axis('off')

        comp_img_path = os.path.join(images_folder, comp_img_name)
        comp_img_name_display = os.path.basename(comp_img_name)
        try:
            comp_img = mpimg.imread(comp_img_path)
        except FileNotFoundError:
            print(f"Error: Compared image not found at {comp_img_path}")
            comp_img = np.zeros(ref_img_size, dtype=np.uint8)
            comp_img_name_display += " (Not Found)"
        except Exception as e:
            print(f"Error loading compared image {comp_img_path}: {e}")
            comp_img = np.zeros(ref_img_size, dtype=np.uint8)
            comp_img_name_display += " (Load Error)"
        
        ax_comp = fig.add_subplot(gs[1 + i, 1])
        ax_comp.imshow(comp_img)
        ax_comp.set_title(f"Compared:\n{comp_img_name_display}", fontsize=9)
        ax_comp.axis('off')

    # --- 5. Customize and Save ---
    fig.suptitle("Image Similarity Analysis Report", fontsize=16, y=0.99) 
    
    # Use constrained_layout for better automatic spacing if GridSpec hspace/wspace isn't perfect
    # plt.tight_layout(rect=[0, 0, 1, 0.96]) # Often conflicts with GridSpec, try constrained_layout
    try:
        fig.set_constrained_layout_pads(w_pad=0.1, h_pad=0.1, hspace=0.05, wspace=0.05) # Experimental
        fig.set_constrained_layout(True)
    except AttributeError: # Older matplotlib might not have set_constrained_layout_pads
        try:
            plt.tight_layout(rect=[0, 0.01, 1, 0.97]) # Make space for suptitle and bottom
        except: # tight_layout can sometimes fail with complex layouts
             print("Warning: tight_layout failed. Spacing might not be optimal.")


    try:
        plt.savefig(output_filename, bbox_inches='tight', dpi=150)
        print(f"Report image saved to {output_filename}")
    except Exception as e:
        print(f"Error saving image: {e}")
    
    plt.close(fig)




warnings.filterwarnings('ignore')


# Define a practical maximum dimension for processing to prevent excessive memory/CPU usage
# and to stay well within Pillow's hard limit of 2^16 - 1 (65535)
# You can adjust this based on your system's capabilities and needs.
# Using Pillow's hard limit directly if such large images are truly necessary for comparison.
PILLOW_MAX_DIM = 2**16 - 1  # 65535
# For more practical purposes, you might want a smaller cap, e.g., 8192 or 16384
PRACTICAL_MAX_DIM = 8192
PROCESS_MAX_DIM = PRACTICAL_MAX_DIM # Choose the effective max dimension

class ImageComparator:
    def __init__(self, reference_image_path):
        img_bytes = np.fromfile(reference_image_path, np.uint8)
        self.reference_image_original_cv = cv2.imdecode(img_bytes, cv2.IMREAD_COLOR) # Store original load
        self.reference_image_path = reference_image_path

        if self.reference_image_original_cv is None:
            raise ValueError(f"Could not load reference image from {reference_image_path}")

        h, w = self.reference_image_original_cv.shape[:2]
        current_max_dim = max(h, w)

        if current_max_dim > PROCESS_MAX_DIM:
            print(f"Warning: Reference image dimension ({w}x{h}) exceeds processing limit ({PROCESS_MAX_DIM}).")
            scale_factor = PROCESS_MAX_DIM / current_max_dim
            
            new_w = int(w * scale_factor)
            new_h = int(h * scale_factor)
            # Ensure new dimensions are at least 1x1
            new_w = max(1, new_w)
            new_h = max(1, new_h)
            
            print(f"Resizing reference image from {w}x{h} to {new_w}x{new_h} for compatibility.")
            # Use INTER_AREA for shrinking, good quality
            self.reference_image = cv2.resize(self.reference_image_original_cv, (new_w, new_h), interpolation=cv2.INTER_AREA)
        else:
            self.reference_image = self.reference_image_original_cv
            
        self.reference_gray = cv2.cvtColor(self.reference_image, cv2.COLOR_BGR2GRAY)
        self.reference_rgb = cv2.cvtColor(self.reference_image, cv2.COLOR_BGR2RGB)
        
        # Cache for loaded images (RGB, for thumbnails)
        self.loaded_images_cache = {}
        ref_cache_key = Path(self.reference_image_path).name + ' (Reference)'
        # Store the (potentially resized) RGB version
        self.loaded_images_cache[ref_cache_key] = self.reference_rgb
        
    def load_and_preprocess_image(self, image_path):
        img_bytes = np.fromfile(image_path, np.uint8)
        img_original_cv = cv2.imdecode(img_bytes, cv2.IMREAD_COLOR)

        if img_original_cv is None:
            raise ValueError(f"Could not load image from {image_path}")
            
        # Target dimensions are from the (potentially resized) reference image
        target_height, target_width = self.reference_image.shape[:2]

        # Check if the original loaded image is excessively large before cv2.resize
        # cv2.resize is generally robust, but good to be aware if it were a bottleneck.
        # The main Pillow error usually happens later (skimage.transform.resize or matplotlib)
        # if the target_width/target_height themselves are too large (due to a large reference).
        # The fix in __init__ should prevent target_width/height from being too large.
        
        # Resize to match (capped) reference image dimensions
        # Use INTER_AREA if shrinking, INTER_CUBIC or INTER_LINEAR if enlarging
        current_h, current_w = img_original_cv.shape[:2]
        if target_width < current_w or target_height < current_h: # Shrinking
            interpolation = cv2.INTER_AREA
        else: # Enlarging or same size
            interpolation = cv2.INTER_CUBIC 
            
        img_resized_cv = cv2.resize(img_original_cv, (target_width, target_height), interpolation=interpolation)
        
        img_gray_resized_cv = cv2.cvtColor(img_resized_cv, cv2.COLOR_BGR2GRAY)
        img_rgb_resized_cv = cv2.cvtColor(img_resized_cv, cv2.COLOR_BGR2RGB)

        # Cache the RGB image (which is now at capped reference size) for thumbnail use
        self.loaded_images_cache[Path(image_path).name] = img_rgb_resized_cv 
        
        return img_resized_cv, img_gray_resized_cv, img_rgb_resized_cv

    def mean_squared_error(self, img1, img2):
        return np.mean((img1.astype(float) - img2.astype(float)) ** 2)
    
    def peak_signal_noise_ratio(self, img1, img2):
        mse = self.mean_squared_error(img1, img2)
        if mse == 0:
            return float('inf')
        max_pixel = 255.0
        return 20 * np.log10(max_pixel / np.sqrt(mse))
    
    def structural_similarity_index(self, img1, img2):
        return metrics.structural_similarity(img1, img2, data_range=255)
    
    def normalized_cross_correlation(self, img1, img2):
        # Ensure inputs are single-channel and float32
        if img1.ndim > 2: img1 = cv2.cvtColor(img1, cv2.COLOR_BGR2GRAY)
        if img2.ndim > 2: img2 = cv2.cvtColor(img2, cv2.COLOR_BGR2GRAY)
        img1_float = img1.astype(np.float32)
        img2_float = img2.astype(np.float32)
        
        # Template matching requires template to be smaller or equal.
        # If they are same size (which they are after preprocessing), result is 1x1 array.
        if img1_float.shape[0] > img2_float.shape[0] or img1_float.shape[1] > img2_float.shape[1]:
             # swap if img1 is larger than img2 (template must be smaller)
             result = cv2.matchTemplate(img1_float, img2_float, cv2.TM_CCORR_NORMED)
        else:
             result = cv2.matchTemplate(img2_float, img1_float, cv2.TM_CCORR_NORMED)
        return np.max(result)
    
    def histogram_correlation(self, img1, img2):
        hist1 = cv2.calcHist([img1], [0], None, [256], [0, 256])
        hist2 = cv2.calcHist([img2], [0], None, [256], [0, 256])
        return cv2.compareHist(hist1, hist2, cv2.HISTCMP_CORREL)
    
    def histogram_chi_square(self, img1, img2):
        hist1 = cv2.calcHist([img1], [0], None, [256], [0, 256])
        hist2 = cv2.calcHist([img2], [0], None, [256], [0, 256])
        chi_square = cv2.compareHist(hist1, hist2, cv2.HISTCMP_CHISQR)
        return 1 / (1 + chi_square + 1e-6) # Add epsilon to avoid division by zero if chi_square is -1
    
    def histogram_intersection(self, img1, img2):
        hist1 = cv2.calcHist([img1], [0], None, [256], [0, 256])
        hist2 = cv2.calcHist([img2], [0], None, [256], [0, 256])
        return cv2.compareHist(hist1, hist2, cv2.HISTCMP_INTERSECT)
    
    def feature_matching_similarity(self, img1, img2):
        orb = cv2.ORB_create()
        kp1, des1 = orb.detectAndCompute(img1, None)
        kp2, des2 = orb.detectAndCompute(img2, None)
        
        if des1 is None or des2 is None or len(kp1) == 0 or len(kp2) == 0: # Added len check
            return 0.0
        
        bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)
        matches = bf.match(des1, des2)
        
        if not matches: # Check if matches list is empty
            return 0.0
            
        matches = sorted(matches, key=lambda x: x.distance)
        good_matches = [m for m in matches if m.distance < 50] # Threshold can be tuned
        
        # Use max(1, ...) to avoid division by zero if no keypoints found.
        similarity = len(good_matches) / max(1, max(len(kp1), len(kp2)))
        return min(similarity, 1.0)
    
    def edge_similarity(self, img1, img2):
        edges1 = cv2.Canny(img1, 100, 200)
        edges2 = cv2.Canny(img2, 100, 200)
        
        intersection = np.logical_and(edges1, edges2).sum()
        union = np.logical_or(edges1, edges2).sum()
        
        if union == 0:
            return 1.0 if intersection == 0 else 0.0 # Both images are blank white/black
        return intersection / union
    
    def gradient_similarity(self, img1, img2):
        grad1_x = cv2.Sobel(img1, cv2.CV_64F, 1, 0, ksize=3)
        grad1_y = cv2.Sobel(img1, cv2.CV_64F, 0, 1, ksize=3)
        # Corrected gradient magnitude calculation: square of sum, then sqrt is not same as sum of squares then sqrt
        grad1_mag = np.sqrt(grad1_x**2 + grad1_y**2) 
        
        grad2_x = cv2.Sobel(img2, cv2.CV_64F, 1, 0, ksize=3)
        grad2_y = cv2.Sobel(img2, cv2.CV_64F, 0, 1, ksize=3)
        grad2_mag = np.sqrt(grad2_x**2 + grad2_y**2)
        
        max_grad1 = np.max(grad1_mag)
        max_grad2 = np.max(grad2_mag)

        # Normalize gradients
        grad1_mag_norm = grad1_mag / (max_grad1 + 1e-8) if max_grad1 > 0 else grad1_mag
        grad2_mag_norm = grad2_mag / (max_grad2 + 1e-8) if max_grad2 > 0 else grad2_mag
        
        # Calculate correlation
        # Ensure no NaNs from flat arrays of zeros
        if np.all(grad1_mag_norm == 0) and np.all(grad2_mag_norm == 0):
            return 1.0 # Both are blank, so perfectly correlated in terms of gradient
        if np.all(grad1_mag_norm == 0) or np.all(grad2_mag_norm == 0):
            return 0.0 # One is blank, other is not

        correlation_matrix = np.corrcoef(grad1_mag_norm.flatten(), grad2_mag_norm.flatten())
        correlation = correlation_matrix[0, 1]
        return correlation if not np.isnan(correlation) else 0.0
    
    def color_histogram_similarity(self, img1_rgb, img2_rgb):
        hist1 = cv2.calcHist([img1_rgb], [0, 1, 2], None, [50, 50, 50], [0, 256, 0, 256, 0, 256])
        hist2 = cv2.calcHist([img2_rgb], [0, 1, 2], None, [50, 50, 50], [0, 256, 0, 256, 0, 256])
        
        cv2.normalize(hist1, hist1, alpha=0, beta=1, norm_type=cv2.NORM_MINMAX)
        cv2.normalize(hist2, hist2, alpha=0, beta=1, norm_type=cv2.NORM_MINMAX)
        
        return cv2.compareHist(hist1, hist2, cv2.HISTCMP_CORREL)
    
    def compare_single_image(self, image_path):
        img, img_gray, img_rgb = self.load_and_preprocess_image(image_path)
        
        results = {
            'Image': Path(image_path).name,
            'MSE': self.mean_squared_error(self.reference_gray, img_gray),
            'PSNR': self.peak_signal_noise_ratio(self.reference_gray, img_gray),
            'SSIM': self.structural_similarity_index(self.reference_gray, img_gray),
            'NCC': self.normalized_cross_correlation(self.reference_gray, img_gray),
            'Hist_Correl': self.histogram_correlation(self.reference_gray, img_gray),
            'Hist_ChiSq': self.histogram_chi_square(self.reference_gray, img_gray),
            'Hist_Intersect': self.histogram_intersection(self.reference_gray, img_gray),
            'Feature_Match': self.feature_matching_similarity(self.reference_gray, img_gray),
            'Edge_Similarity': self.edge_similarity(self.reference_gray, img_gray),
            'Gradient_Sim': self.gradient_similarity(self.reference_gray, img_gray),
            'Color_Hist': self.color_histogram_similarity(self.reference_rgb, img_rgb)
        }
        return results
    
    def compare_multiple_images(self, image_paths):
        results = []
        ref_name = Path(self.reference_image_path).name
        ref_results = {
            'Image': ref_name,
            'MSE': 0.0,
            'PSNR': float('inf'),
            'SSIM': 1.0,
            'NCC': 1.0,
            'Hist_Correl': 1.0,
            'Hist_ChiSq': 1.0,
            'Hist_Intersect': np.sum(cv2.calcHist([self.reference_gray], [0], None, [256], [0, 256])),
            'Feature_Match': 1.0,
            'Edge_Similarity': 1.0,
            'Gradient_Sim': 1.0,
            'Color_Hist': 1.0
        }
        results.append(ref_results)
        # self.loaded_images_cache[ref_name] already set in __init__
        
        for image_path in image_paths:
            try:
                result = self.compare_single_image(image_path)
                results.append(result)
                # print(f"Processed: {Path(image_path).name}")
            except Exception as e:
                print(f"Error processing {image_path}: {str(e)}")
                # Add a placeholder if an image fails, so df columns align
                results.append({'Image': Path(image_path).name + " (Error)", **{k: np.nan for k in ref_results if k != 'Image'}})
                self.loaded_images_cache[Path(image_path).name + " (Error)"] = np.zeros((64,64,3), dtype=np.uint8) # Placeholder image
                continue
        
        return pd.DataFrame(results)
    
    def create_similarity_heatmap(self, df, save_path=None, thumbnail_size=(400, 50), thumbnail_zoom=1.0):
        if df.empty:
            print("DataFrame is empty. Cannot generate heatmap.")
            return

        numeric_cols = df.select_dtypes(include=[np.number]).columns
        heatmap_data = df[numeric_cols].copy()
        
        if 'MSE' in heatmap_data.columns:
            valid_mse = heatmap_data['MSE'][np.isfinite(heatmap_data['MSE'])]
            max_mse = valid_mse.max() if not valid_mse.empty else 1.0
            heatmap_data['MSE_Norm'] = 1 - (heatmap_data['MSE'] / (max_mse + 1e-9)) 
            heatmap_data['MSE_Norm'] = heatmap_data['MSE_Norm'].fillna(0) 
            heatmap_data = heatmap_data.drop('MSE', axis=1)
        
        if 'PSNR' in heatmap_data.columns:
            psnr_finite = heatmap_data['PSNR'][np.isfinite(heatmap_data['PSNR'])]
            if not psnr_finite.empty:
                max_psnr = psnr_finite.max() 
                min_psnr = psnr_finite.min() 
                
                if max_psnr > min_psnr:
                    heatmap_data['PSNR_Norm'] = (heatmap_data['PSNR'] - min_psnr) / (max_psnr - min_psnr)
                else: 
                     heatmap_data['PSNR_Norm'] = heatmap_data['PSNR'].apply(lambda x: 1.0 if x == float('inf') else (0.5 if pd.notna(x) else 0.0))
                heatmap_data['PSNR_Norm'] = heatmap_data['PSNR_Norm'].replace(float('inf'), 1.0) 
                heatmap_data['PSNR_Norm'] = heatmap_data['PSNR_Norm'].fillna(0)
            else: 
                heatmap_data['PSNR_Norm'] = heatmap_data['PSNR'].apply(lambda x: 1.0 if x == float('inf') else 0.0)
            heatmap_data = heatmap_data.drop('PSNR', axis=1)
        
        if 'Hist_Intersect' in heatmap_data.columns:
            valid_intersect = heatmap_data['Hist_Intersect'][np.isfinite(heatmap_data['Hist_Intersect'])]
            max_intersect = valid_intersect.max() if not valid_intersect.empty else 1.0
            heatmap_data['Hist_Intersect_Norm'] = heatmap_data['Hist_Intersect'] / (max_intersect + 1e-9)
            heatmap_data['Hist_Intersect_Norm'] = heatmap_data['Hist_Intersect_Norm'].fillna(0)
            heatmap_data = heatmap_data.drop('Hist_Intersect', axis=1)
        
        fig, ax = plt.subplots(figsize=(max(2, len(df)*1.5), 3)) 
        
        sns.heatmap(heatmap_data.T, 
                   annot=True, 
                   fmt='.2f', 
                   cmap='RdYlGn', 
                   center=0.5,
                   cbar_kws={'label': 'Similarity Score'},
                   ax=ax)
        
        ax.set_title('Image Similarity Comparison Heatmap', fontsize=16)
        ax.set_xlabel('Images', fontsize=14)
        ax.set_ylabel('Similarity Metrics', fontsize=14)
        ax.set_xticklabels([]) 
        ax.tick_params(axis='x', length=0) 

        offset_points = 0

        for i, img_name_in_df in enumerate(df['Image']):
            img_data = self.loaded_images_cache.get(img_name_in_df)
            if img_data is None:
                print(f"Warning: Image data for '{img_name_in_df}' not found in cache for heatmap.")
                img_data = np.zeros((*thumbnail_size, 3), dtype=np.uint8) # thumbnail_size is (height, width)
            
            # ADD THIS DEBUG LINE:
            # print(f"Thumbnailing '{img_name_in_df}': img_data.shape={img_data.shape}, img_data.dtype={img_data.dtype}, thumbnail_size={thumbnail_size}")

            if img_data.dtype == np.float32 or img_data.dtype == np.float64:
                img_data = (np.clip(img_data, 0, 1) * 255).astype(np.uint8) if img_data.max() <=1 else img_data.astype(np.uint8)

            # thumb = skimage_resize(img_data, thumbnail_size, preserve_range=True, anti_aliasing=True).astype(np.uint8)
            thumb = skimage_resize(img_data, thumbnail_size, preserve_range=True, anti_aliasing=False).astype(np.uint8)
            
            imagebox = OffsetImage(thumb, zoom=thumbnail_zoom)
            ab = AnnotationBbox(offsetbox=imagebox, 
                                xy=(i + 0.5, -.5), 
                                xybox=(0., offset_points), 
                                frameon=False,
                                xycoords=ax.get_xaxis_transform(), 
                                boxcoords="offset points",
                                pad=0.0,
                                annotation_clip=False) 
            ax.add_artist(ab)

            # text_y_offset_in_points = offset_points - (thumbnail_size[0] * thumbnail_zoom / 2) - 15 # 15 is padding

            # ax.annotate(Path(img_name_in_df).name,
            #             xy=(i + 0.5, 0),  # Anchor point: X in data coords, Y at the x-axis line (0 in axes fraction)
            #             xycoords=ax.get_xaxis_transform(),
            #             xytext=(0, text_y_offset_in_points),  # Offset from anchor point xy
            #             textcoords='offset points',  # Interprets xytext in points
            #             ha='center',
            #             va='top',
            #             fontsize=8,
            #             rotation=45,
            #             annotation_clip=False, # Allow text to be outside axes if necessary
            #             bbox=dict(facecolor='white', alpha=0.5, pad=0.1, boxstyle='round,pad=0.2') if "Error" in img_name_in_df else None
            #            )

        plt.subplots_adjust(bottom=0.25 if len(df['Image']) < 7 else 0.35) 
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.show()
    
    def generate_summary_report(self, df):
        print("="*80)
        print("IMAGE SIMILARITY COMPARISON REPORT")
        print("="*80)
        print(f"Reference Image: {Path(self.reference_image_path).name}")
        print(f"Total Images Compared: {len(df) - 1}")
        print("\nMETRIC DESCRIPTIONS:")
        print("-"*40)
        print("MSE_Norm: Normalized Mean Squared Error (higher = more similar)")
        print("PSNR_Norm: Normalized Peak Signal-to-Noise Ratio (higher = more similar)")
        print("SSIM: Structural Similarity Index (higher = more similar)")
        print("NCC: Normalized Cross Correlation (higher = more similar)")
        print("Hist_Correl: Histogram Correlation (higher = more similar)")
        print("Hist_ChiSq: Histogram Chi-Square Similarity (higher = more similar)")
        print("Hist_Intersect_Norm: Normalized Histogram Intersection (higher = more similar)")
        print("Feature_Match: Feature Matching Similarity (higher = more similar)")
        print("Edge_Similarity: Edge-based Similarity (higher = more similar)")
        print("Gradient_Sim: Gradient Similarity (higher = more similar)")
        print("Color_Hist: Color Histogram Similarity (higher = more similar)")
        print("\n" + "="*80)
