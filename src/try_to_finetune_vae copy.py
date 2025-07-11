import torch
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from PIL import Image, ImageDraw
import os
from tqdm import tqdm # For progress bars
from transformers import AutoImageProcessor # Correctly from transformers
from diffusers import AutoencoderKL      # Correctly from diffusers
from accelerate import Accelerator # For easier device management and mixed precision
import torchvision.utils as vutils

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

# Ensure output directories exist
os.makedirs(OUTPUT_DIR, exist_ok=True)
os.makedirs(COLLAGE_DIR, exist_ok=True)


# --- 1. Custom Dataset ---
class CustomImageDataset(Dataset):
    def __init__(self, image_dir, image_processor, target_size=(256, 256)):
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

# --- Helper to collate batches and filter Nones ---
def collate_fn(batch):
    batch = [item for item in batch if item[0] is not None]
    if not batch:
        return None, None
    processed_images = torch.stack([item[0] for item in batch])
    original_images = [item[1] for item in batch]
    return processed_images, original_images


# --- 2. Initialize Accelerator, Model, Processor, Optimizer ---
accelerator = Accelerator(mixed_precision="fp16" if torch.cuda.is_available() else "no")
device = accelerator.device

# Load processor from a base SD model
try:
    image_processor = AutoImageProcessor.from_pretrained(
        PREPROCESSOR_MODEL_ID, 
        subfolder="feature_extractor", 
        revision=None
    )   # Use subfolder for older models if needed
    # For newer models, `subfolder` might not be needed, or it could be different.
    # If the above fails, try without subfolder or check the specific model's Hub page.
    # e.g., image_processor = AutoImageProcessor.from_pretrained(PREPROCESSOR_MODEL_ID)
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
# This should ideally come from the image_processor now
if hasattr(image_processor, 'size'):
    if isinstance(image_processor.size, dict):
        # Common for newer processors: {'shortest_edge': 512} or {'height': 512, 'width': 512}
        img_size = image_processor.size.get('height', image_processor.size.get('shortest_edge'))
        if img_size is None: # Fallback if structure is different
             img_size = image_processor.size.get('width', 512)
    elif isinstance(image_processor.size, (int, float)): # Older style, just an int
        img_size = int(image_processor.size)
    else: # Default if unsure
        img_size = 512
    vae_input_size = (img_size, img_size)
elif hasattr(image_processor, 'crop_size'): # Another common attribute
     if isinstance(image_processor.crop_size, dict):
         img_size = image_processor.crop_size.get('height', 512)
     else:
         img_size = image_processor.crop_size
     vae_input_size = (img_size, img_size)
else:
    vae_input_size = (512, 512) # Default for many SD VAEs
    print(f"Warning: Could not reliably infer VAE input size from processor. Using default {vae_input_size}.")
print(f"Determined VAE input size: {vae_input_size}")


# Create dataset and dataloader
train_dataset = CustomImageDataset(IMAGE_DIR, image_processor, target_size=vae_input_size)
if len(train_dataset) == 0:
    print(f"No images found in {IMAGE_DIR}. Please check the path and image extensions.")
    exit()

train_dataloader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, collate_fn=collate_fn)

optimizer = torch.optim.AdamW(vae.parameters(), lr=LEARNING_RATE)

# Prepare everything with Accelerator
vae, optimizer, train_dataloader = accelerator.prepare(
    vae, optimizer, train_dataloader
)

# --- Helper function to save collages ---
def save_collage(original_images, reconstructed_images, epoch, step, filename="reconstruction_collage.png"):
    """Saves a collage of original and reconstructed images."""
    num_images = len(original_images)
    rows = 2  # One row for originals, one for reconstructions

    # Create a new image with enough space for all images
    img_width, img_height = original_images[0].size
    collage_width = num_images * img_width
    collage_height = rows * img_height
    collage = Image.new("RGB", (collage_width, collage_height))

    # Paste original images
    for i, img in enumerate(original_images):
        collage.paste(img, (i * img_width, 0))  # Top row

    # Convert reconstructed images to PIL Images
    reconstructed_pil_images = []
    for tensor_img in reconstructed_images:
        # Move to CPU, detach, and normalize if necessary
        tensor_img = tensor_img.cpu().detach()
        # The images are probably normalized, so unnormalize to [0, 1] range
        tensor_img = (tensor_img + 1) / 2  # Assuming input is scaled to [-1, 1]
        tensor_img = torch.clamp(tensor_img, 0, 1)

        # Convert to PIL Image
        img_grid = tensor_img.unsqueeze(0)  # Add batch dimension
        grid = vutils.make_grid(img_grid, padding=0, normalize=False) # Create the grid
        ndarr = grid.mul(255).add_(0.5).clamp_(0, 255).permute(1, 2, 0).to("cpu", torch.uint8).numpy()
        pil_img = Image.fromarray(ndarr) # ndarr to pillow

        reconstructed_pil_images.append(pil_img)

    # Paste reconstructed images
    for i, img in enumerate(reconstructed_pil_images):
        collage.paste(img, (i * img_width, img_height))  # Bottom row

    # Add titles to the rows
    draw = ImageDraw.Draw(collage)
    draw.text((10, 10), "Original", fill=(255, 255, 255))
    draw.text((10, img_height + 10), "Reconstructed", fill=(255, 255, 255))


    filepath = os.path.join(COLLAGE_DIR, f"collage_epoch_{epoch+1}_step_{step}.png")
    collage.save(filepath)
    # accelerator.print(f"Saved collage to {filepath}")


print(f"Starting fine-tuning for {NUM_EPOCHS} epochs on {len(train_dataset)} images.")
print(f"Device: {device}")
print(f"VAE input size: {vae_input_size}")
print(f"Batch size: {BATCH_SIZE}, Learning rate: {LEARNING_RATE}")


# --- 3. Training Loop ---
for epoch in range(NUM_EPOCHS):
    vae.train()
    total_loss = 0
    progress_bar = tqdm(train_dataloader, desc=f"Epoch {epoch+1}/{NUM_EPOCHS}", disable=not accelerator.is_local_main_process)

    for step, (batch, original_images) in enumerate(progress_bar):
        if batch is None: # Skip if batch is empty after filtering Nones
            continue

        with accelerator.accumulate(vae): # Handles gradient accumulation if enabled
            # pixel_values = batch.to(device) # Accelerator handles device placement
            pixel_values = batch # Already on device thanks to accelerator.prepare(train_dataloader)

            # Forward pass: encode, sample, decode
            # The AutoencoderKL's forward pass does this:
            # 1. encode input -> get latent distribution (mean, logvar)
            # 2. sample from this distribution -> get latents
            # 3. decode latents -> get reconstructed image
            # The forward pass itself doesn't directly return mean and logvar for KL loss,
            # so we might need to call encode and decode separately if we want full VAE loss.
            # However, for fine-tuning, often just reconstruction loss is used, or the VAE's
            # own `forward` method might handle KL internally if it's a true VAE implementation
            # being trained from scratch.
            # For fine-tuning a pre-trained one, we'll explicitly get the distribution.

            encoder_output = vae.encode(pixel_values) # vae is potentially accelerator.unwrap_model(vae)
            latent_dist = encoder_output.latent_dist
            latents = latent_dist.sample() # Sample from the distribution

            # Optional: Apply scaling factor if your VAE uses one (common in diffusion)
            if hasattr(vae.config, 'scaling_factor') and vae.config.scaling_factor is not None:
                latents = latents * vae.config.scaling_factor
            else:
                # Fallback for SD VAEs if not in config (use with caution)
                # latents = latents * 0.18215
                pass


            reconstructed_pixels = vae.decode(latents).sample

            # Calculate reconstruction loss (e.g., MSE)
            reconstruction_loss = F.mse_loss(reconstructed_pixels, pixel_values, reduction="mean")

            # Calculate KL divergence
            # kl_div = -0.5 * torch.sum(1 + latent_dist.logvar - latent_dist.mean.pow(2) - latent_dist.logvar.exp(), dim=[1,2,3])
            # kl_loss = torch.mean(kl_div)
            # A simpler way using the provided kl method in DiagonalGaussianDistribution
            kl_loss = latent_dist.kl().mean() # .mean() to average over batch

            # Total loss
            loss = reconstruction_loss + KL_WEIGHT * kl_loss

            accelerator.backward(loss)
            optimizer.step()
            optimizer.zero_grad()

        total_loss += loss.item()
        progress_bar.set_postfix({"Loss": loss.item(), "Recon Loss": reconstruction_loss.item(), "KL Loss": kl_loss.item()})

        # Save reconstruction collages
        if (epoch + 1) % SAVE_COLLAGES_EVERY_N_EPOCHS == 0 and original_images is not None:
            # Move reconstructed_pixels to CPU and detach from the graph
            reconstructed_pixels_cpu = [img.to("cpu").detach() for img in reconstructed_pixels]

            # Save a collage of original and reconstructed images
            save_collage(original_images, reconstructed_pixels_cpu, epoch, step)

    avg_loss = total_loss / len(train_dataloader)
    accelerator.print(f"Epoch {epoch+1} Average Loss: {avg_loss:.4f}")

    # Save model checkpoint
    if (epoch + 1) % SAVE_EVERY_N_EPOCHS == 0 or (epoch + 1) == NUM_EPOCHS:
        if accelerator.is_main_process:
            unwrapped_model = accelerator.unwrap_model(vae)
            save_path = os.path.join(OUTPUT_DIR, f"vae_epoch_{epoch+1}")
            unwrapped_model.save_pretrained(save_path)
            image_processor.save_pretrained(save_path) # Save processor too
            accelerator.print(f"Saved model checkpoint to {save_path}")

accelerator.print("Fine-tuning complete.")