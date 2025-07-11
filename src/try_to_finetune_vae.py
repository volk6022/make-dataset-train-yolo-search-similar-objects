import torch
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader, random_split
from torchvision import transforms
from PIL import Image, ImageDraw
import os
from tqdm import tqdm # For progress bars
from transformers import AutoImageProcessor # Correctly from transformers
from diffusers import AutoencoderKL      # Correctly from diffusers
from accelerate import Accelerator # For easier device management and mixed precision
import torchvision.utils as vutils
# from google.colab import drive
import numpy as np
import shutil

# Assuming drive is already mounted from previous code
# drive.mount('/content/drive')

# --- Configuration ---
IMAGE_DIR = "cropped_objects - Copy/" # CHANGE THIS
VAE_MODEL_ID = "stabilityai/sd-vae-ft-mse" # Or another VAE
OUTPUT_DIR = "./finetuned_vae" # Directory to save the fine-tuned model
PREPROCESSOR_MODEL_ID = "CompVis/stable-diffusion-v1-4" # Or "runwayml/stable-diffusion-v1-5" or another SD base
NUM_EPOCHS = 10
BATCH_SIZE = 2 # Adjust based on your GPU memory
LEARNING_RATE = 1e-5
KL_WEIGHT = 1e-6 # Weight for the KL divergence term in the loss
SAVE_EVERY_N_EPOCHS = 1 # How often to save checkpoints
COLLAGE_DIR = "./reconstructions"  # Directory to save collages
SAVE_COLLAGES_EVERY_N_EPOCHS = 1
VALIDATION_SPLIT = 0.1 # Fraction of data to use for validation

# Ensure output directories exist
os.makedirs(OUTPUT_DIR, exist_ok=True)
os.makedirs(COLLAGE_DIR, exist_ok=True)

# --- 1. Custom Dataset (from previous code) ---
class CustomImageDataset(Dataset):
    def __init__(self, image_dir, image_processor, target_size=(224, 224)):
        self.image_dir = image_dir
        self.image_paths = [os.path.join(image_dir, fname) for fname in os.listdir(image_dir)
                            if fname.lower().endswith(('.png', '.jpg', '.jpeg'))]
        self.image_processor = image_processor
        self.target_size = target_size # VAEs often expect square inputs

        # Custom transform to handle non-square images by padding
        self.custom_transform = transforms.Compose([
            lambda img: self.pad_to_square(img, 0), # Pad with black
            transforms.Resize(self.target_size), # Resize to VAE's expected input
            # Note: AutoImageProcessor will handle ToTensor and normalization
        ])

    def pad_to_square(self, img, padding_value=0):
        # Pad to square, then resize. This helps with non-square inputs like 50x400
        w, h = img.size
        max_wh = max(w, h)
        hp = int((max_wh - w) / 2)
        vp = int((max_wh - h) / 2)
        padding = (hp, vp, hp, vp) # left, top, right, bottom
        return transforms.functional.pad(img, padding, padding_value, 'constant')

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        img_path = self.image_paths[idx]
        try:
            image = Image.open(img_path).convert("RGB")
            # Apply custom padding and resizing *before* the AutoImageProcessor
            image = self.custom_transform(image)
            # AutoImageProcessor handles final ToTensor and normalization
            processed_image = self.image_processor(image, return_tensors="pt").pixel_values
            return processed_image.squeeze(0), image  # Return processed image and original PIL image
        except Exception as e:
            print(f"Error loading or processing image {img_path}: {e}")
            # Return a dummy tensor or skip; here we'll return None and filter later
            return None, None

# --- Helper to collate batches and filter Nones (from previous code) ---
def collate_fn(batch):
    batch = [item for item in batch if item[0] is not None]
    if not batch:
        return None, None
    processed_images = torch.stack([item[0] for item in batch])
    original_images = [item[1] for item in batch]
    return processed_images, original_images

# --- 2. Initialize Accelerator, Model, Processor, Optimizer (from previous code) ---
accelerator = Accelerator(mixed_precision="fp16" if torch.cuda.is_available() else "no")
device = accelerator.device

# Load processor from a base SD model
try:
    image_processor = AutoImageProcessor.from_pretrained(
        PREPROCESSOR_MODEL_ID,
        subfolder="feature_extractor",
        revision=None,
    )
except OSError as e:
    print(f"Trying to load image_processor without subfolder for {PREPROCESSOR_MODEL_ID} due to: {e}")
    try:
        image_processor = AutoImageProcessor.from_pretrained(PREPROCESSOR_MODEL_ID)
    except Exception as e_inner:
        print(f"Could not load AutoImageProcessor for {PREPROCESSOR_MODEL_ID}. Error: {e_inner}")
        print("Please ensure you have a valid model ID that contains a preprocessor_config.json")
        exit()

# Load VAE model
try:
    vae = AutoencoderKL.from_pretrained(VAE_MODEL_ID) # This is from diffusers
except Exception as e:
    print(f"Could not load VAE model {VAE_MODEL_ID}. Error: {e}")
    exit()

# Prepare VAE input size
if hasattr(image_processor, 'size'):
    if isinstance(image_processor.size, dict):
        img_size = image_processor.size.get('height', image_processor.size.get('shortest_edge'))
        if img_size is None:
             img_size = image_processor.size.get('width', 512)
    elif isinstance(image_processor.size, (int, float)):
        img_size = int(image_processor.size)
    else:
        img_size = 512
    vae_input_size = (img_size, img_size)
elif hasattr(image_processor, 'crop_size'):
     if isinstance(image_processor.crop_size, dict):
         img_size = image_processor.crop_size.get('height', 512)
     else:
         img_size = image_processor.crop_size
     vae_input_size = (img_size, img_size)
else:
    vae_input_size = (512, 512) # Default for many SD VAEs
    print(f"Warning: Could not reliably infer VAE input size from processor. Using default {vae_input_size}.")
print(f"Determined VAE input size: {vae_input_size}")


# --- Split Dataset into Train and Validation ---
full_dataset = CustomImageDataset(IMAGE_DIR, image_processor, target_size=vae_input_size)
if len(full_dataset) == 0:
    print(f"No images found in {IMAGE_DIR}. Please check the path and image extensions.")
    exit()

train_size = int((1 - VALIDATION_SPLIT) * len(full_dataset))
val_size = len(full_dataset) - train_size
train_dataset, val_dataset = random_split(full_dataset, [train_size, val_size])

print(f"Total images: {len(full_dataset)}")
print(f"Training images: {len(train_dataset)}")
print(f"Validation images: {len(val_dataset)}")

# Create dataloaders
train_dataloader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, collate_fn=collate_fn)
val_dataloader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False, collate_fn=collate_fn)

optimizer = torch.optim.AdamW(vae.parameters(), lr=LEARNING_RATE)

# Prepare everything with Accelerator
vae, optimizer, train_dataloader, val_dataloader = accelerator.prepare(
    vae, optimizer, train_dataloader, val_dataloader
)

# --- Helper function to save collages (from previous code) ---
def save_collage(original_images, reconstructed_images, epoch, step, is_validation=False, filename="reconstruction_collage.png"):
    """Saves a collage of original and reconstructed images."""
    if not original_images or not reconstructed_images:
        return

    num_images = len(original_images)
    rows = 2  # One row for originals, one for reconstructions

    # Create a new image with enough space for all images
    # Ensure dimensions are consistent - use the size from original PIL images
    # If original_images list is empty, skip. If only some are None, handle.
    valid_original_images = [img for img in original_images if img is not None]
    if not valid_original_images:
        # print("Warning: No valid original images to create collage.")
        return

    img_width, img_height = valid_original_images[0].size
    collage_width = num_images * img_width
    collage_height = rows * img_height
    collage = Image.new("RGB", (collage_width, collage_height))

    # Paste original images
    for i, img in enumerate(original_images):
        if img is not None:
             collage.paste(img, (i * img_width, 0))  # Top row

    # Convert reconstructed images to PIL Images
    reconstructed_pil_images = []
    for tensor_img in reconstructed_images:
        # Move to CPU, detach
        tensor_img = tensor_img.cpu().detach()
        # The images are probably normalized [-1, 1], unnormalize to [0, 1]
        tensor_img = (tensor_img + 1) / 2
        tensor_img = torch.clamp(tensor_img, 0, 1)

        # Convert to PIL Image using torchvision.utils.make_grid
        # Add batch dimension if not present
        if tensor_img.ndim == 3:
            tensor_img = tensor_img.unsqueeze(0)

        try:
            grid = vutils.make_grid(tensor_img, padding=0, normalize=False)
            # grid is [C, H, W], convert to [H, W, C] for numpy
            ndarr = grid.mul(255).add_(0.5).clamp_(0, 255).permute(1, 2, 0).to("cpu", torch.uint8).numpy()
            pil_img = Image.fromarray(ndarr)
            reconstructed_pil_images.append(pil_img)
        except Exception as e:
            print(f"Error converting tensor to PIL for collage: {e}")
            # Add a blank image placeholder or skip
            blank_img = Image.new("RGB", (img_width, img_height), color = 'red') # Red for error
            reconstructed_pil_images.append(blank_img)


    # Paste reconstructed images
    for i, img in enumerate(reconstructed_pil_images):
        collage.paste(img, (i * img_width, img_height))  # Bottom row

    # Add titles to the rows
    draw = ImageDraw.Draw(collage)
    draw.text((10, 10), "Original", fill=(255, 255, 255))
    draw.text((10, img_height + 10), "Reconstructed", fill=(255, 255, 255))

    prefix = "val" if is_validation else "train"
    filepath = os.path.join(COLLAGE_DIR, f"{prefix}_collage_epoch_{epoch+1}_step_{step}.png")
    collage.save(filepath)
    # accelerator.print(f"Saved collage to {filepath}")


print(f"Starting fine-tuning for {NUM_EPOCHS} epochs.")
print(f"Device: {device}")
print(f"VAE input size: {vae_input_size}")
print(f"Train Batch size: {BATCH_SIZE}, Validation Batch size: {BATCH_SIZE}")
print(f"Learning rate: {LEARNING_RATE}")
print(f"Validation split: {VALIDATION_SPLIT}")


# Clear the collage directory at the beginning of each epoch
if os.path.exists(COLLAGE_DIR):
    shutil.rmtree(COLLAGE_DIR)  # Remove the directory and its contents
os.makedirs(COLLAGE_DIR, exist_ok=True)  # Recreate the directory

# --- 3. Training Loop ---
for epoch in range(NUM_EPOCHS):
    # --- Training Phase ---
    vae.train()
    train_total_loss = 0
    progress_bar = tqdm(train_dataloader, desc=f"Epoch {epoch+1}/{NUM_EPOCHS} (Train)", disable=not accelerator.is_local_main_process)

    for step, (batch, original_images) in enumerate(progress_bar):
        if batch is None: # Skip if batch is empty after filtering Nones
            continue

        with accelerator.accumulate(vae): # Handles gradient accumulation if enabled
            pixel_values = batch # Already on device thanks to accelerator.prepare(train_dataloader)

            encoder_output = vae.encode(pixel_values)
            latent_dist = encoder_output.latent_dist
            latents = latent_dist.sample()

            # Optional: Apply scaling factor if your VAE uses one (common in diffusion)
            if hasattr(vae.config, 'scaling_factor') and vae.config.scaling_factor is not None:
                 latents = latents * vae.config.scaling_factor
            # else:
                 # If scaling factor is needed but not in config, uncomment and use a known value
                 # latents = latents * 0.18215

            reconstructed_pixels = vae.decode(latents).sample

            # Calculate reconstruction loss (e.g., MSE)
            reconstruction_loss = F.mse_loss(reconstructed_pixels, pixel_values, reduction="mean")

            # Calculate KL divergence
            kl_loss = latent_dist.kl().mean()

            # Total loss
            loss = reconstruction_loss + KL_WEIGHT * kl_loss

            accelerator.backward(loss)
            optimizer.step()
            optimizer.zero_grad()

        train_total_loss += loss.item()
        progress_bar.set_postfix({"Loss": loss.item(), "Recon Loss": reconstruction_loss.item(), "KL Loss": kl_loss.item()})

        # Save reconstruction collages during training (optional)
        if (epoch + 1) % SAVE_COLLAGES_EVERY_N_EPOCHS == 0 and original_images is not None and accelerator.is_main_process:
             # For frequent saving, maybe only save on the first step or periodically
             if step == 0 or (step + 1) % 100 == 0: # Example: save first batch and every 100 steps
                reconstructed_pixels_cpu = [img.to("cpu").detach() for img in reconstructed_pixels]
                save_collage(original_images, reconstructed_pixels_cpu, epoch, step, is_validation=False)

    avg_train_loss = train_total_loss / len(train_dataloader)
    accelerator.print(f"Epoch {epoch+1} Training Average Loss: {avg_train_loss:.4f}")

    # --- Validation Phase ---
    vae.eval()
    val_total_loss = 0
    val_progress_bar = tqdm(val_dataloader, desc=f"Epoch {epoch+1}/{NUM_EPOCHS} (Val)", disable=not accelerator.is_local_main_process)
    val_original_images_sample = None
    val_reconstructed_images_sample = None

    with torch.no_grad():
        for step, (batch, original_images) in enumerate(val_progress_bar):
            if batch is None:
                continue

            pixel_values = batch # Already on device

            encoder_output = vae.encode(pixel_values)
            latent_dist = encoder_output.latent_dist
            latents = latent_dist.sample()

            if hasattr(vae.config, 'scaling_factor') and vae.config.scaling_factor is not None:
                 latents = latents * vae.config.scaling_factor
            # else:
                 # latents = latents * 0.18215

            reconstructed_pixels = vae.decode(latents).sample

            reconstruction_loss = F.mse_loss(reconstructed_pixels, pixel_values, reduction="mean")
            kl_loss = latent_dist.kl().mean()
            loss = reconstruction_loss + KL_WEIGHT * kl_loss

            val_total_loss += loss.item()
            val_progress_bar.set_postfix({"Loss": loss.item(), "Recon Loss": reconstruction_loss.item(), "KL Loss": kl_loss.item()})

            # Save a sample validation collage (e.g., from the first batch)
            if step == 0 and original_images is not None and accelerator.is_main_process:
                val_original_images_sample = original_images
                val_reconstructed_images_sample = [img.to("cpu").detach() for img in reconstructed_pixels]


    avg_val_loss = val_total_loss / len(val_dataloader)
    accelerator.print(f"Epoch {epoch+1} Validation Average Loss: {avg_val_loss:.4f}")

    # Save validation collage for the epoch
    if (epoch + 1) % SAVE_COLLAGES_EVERY_N_EPOCHS == 0 and val_original_images_sample is not None and accelerator.is_main_process:
         save_collage(val_original_images_sample, val_reconstructed_images_sample, epoch, 0, is_validation=True)


    # Save model checkpoint
    if (epoch + 1) % SAVE_EVERY_N_EPOCHS == 0 or (epoch + 1) == NUM_EPOCHS:
        if accelerator.is_main_process:
            unwrapped_model = accelerator.unwrap_model(vae)
            save_path = os.path.join(OUTPUT_DIR, f"vae_epoch_{epoch+1}")
            unwrapped_model.save_pretrained(save_path)
            image_processor.save_pretrained(save_path) # Save processor too
            accelerator.print(f"Saved model checkpoint to {save_path}")

accelerator.print("Fine-tuning complete.")