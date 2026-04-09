import torch.cuda
from PIL import Image
import cv2
import torch
from torchvision import transforms



target_size=(224, 224)

def pad_to_square(img, padding_value=0):
    w, h = img.size
    max_wh = max(w, h)
    hp = int((max_wh - w) / 2)
    vp = int((max_wh - h) / 2)
    padding = (hp, vp, hp, vp) # left, top, right, bottom
    return transforms.functional.pad(img, padding, padding_value, 'constant')

def pad_to_square_and_resize(img, padding_value=0):
    w, h = img.size
    max_wh = max(w, h)
    hp = int((max_wh - w) / 2)
    vp = int((max_wh - h) / 2)
    padding = (hp, vp, hp, vp) # left, top, right, bottom
    img = transforms.functional.pad(img, padding, padding_value, 'constant')
    return transforms.functional.resize(img, target_size)

def get_device():
    """Gets the appropriate device (CUDA or CPU) for torch."""
    return "cuda" if torch.cuda.is_available() else "cpu"

models = {
    'vit_google': 'google/vit-base-patch16-224-in21k',
    'dino': 'facebook/dinov2-large',
    'dino_giant': 'facebook/dinov2-giant',
    'clip': 'openai/clip-vit-large-patch14',
    'siglip': 'google/siglip-so400m-patch14-384',
    'radio': 'nvidia/RADIO',
    'sd-vae': {
        # 'VAE_MODEL_ID': "stabilityai/sd-vae-ft-mse",
        'VAE_MODEL_ID': "finetuned_vae/vae_epoch_5",
        # 'PREPROCESSOR_MODEL_ID': "CompVis/stable-diffusion-v1-4",
        'PREPROCESSOR_MODEL_ID': "finetuned_vae/vae_epoch_5",
        'device': get_device(),
        'transform': transforms.Compose([
            lambda img: pad_to_square(img, 0), # Pad with black
            transforms.Resize(target_size), # Resize to VAE's expected input
            # Note: AutoImageProcessor will handle ToTensor and normalization
        ]),
    }
}



def init_embedding(embedding_model_name='vit_google', device='cuda'):
    if device == 'cuda' and get_device() == 'cpu':
        print(f"WARNING: device {device} is not aveilable, using 'cpu' as device")
        device = 'cpu'
    if embedding_model_name not in models.keys():
        print(f"WARNING: device {embedding_model_name} is not aveilable, using 'vit_google' as embedding_model_name")
        embedding_model_name='vit_google'
    try:
        if embedding_model_name == 'vit_google':
            from transformers import AutoImageProcessor, AutoModel
            image_processor = AutoImageProcessor.from_pretrained(models[embedding_model_name])
            embedding_model = AutoModel.from_pretrained(models[embedding_model_name]).to(device)
            
        elif embedding_model_name == 'dino':
            from transformers import AutoImageProcessor, AutoModel
            image_processor = AutoImageProcessor.from_pretrained(models[embedding_model_name], use_fast=True)
            embedding_model = AutoModel.from_pretrained(models[embedding_model_name]).to(device)
            embedding_model.eval()

        elif embedding_model_name == 'dino_giant':
            from transformers import AutoImageProcessor, AutoModel
            image_processor = AutoImageProcessor.from_pretrained(models[embedding_model_name], use_fast=True)
            embedding_model = AutoModel.from_pretrained(models[embedding_model_name]).to(device)
            embedding_model.eval()

        elif embedding_model_name == 'clip':
            from transformers import CLIPVisionModel, CLIPProcessor
            image_processor = CLIPProcessor.from_pretrained(models[embedding_model_name])
            embedding_model = CLIPVisionModel.from_pretrained(models[embedding_model_name]).to(device)
            embedding_model.eval()

        elif embedding_model_name == 'siglip':
            from transformers import SiglipVisionModel, AutoProcessor
            image_processor = AutoProcessor.from_pretrained(models[embedding_model_name])
            embedding_model = SiglipVisionModel.from_pretrained(models[embedding_model_name]).to(device)
            embedding_model.eval()

        elif embedding_model_name == 'radio':
            import timm
            radio_model = timm.create_model('hf_hub:nvidia/RADIO', pretrained=True).to(device)
            radio_model.eval()
            data_config = timm.data.resolve_model_data_config(radio_model)
            radio_transforms = timm.data.create_transform(**data_config, is_training=False)
            # Return transforms as image_processor (callable: PIL -> Tensor)
            return radio_transforms, radio_model, device

        elif embedding_model_name == 'sd-vae':
            models[embedding_model_name]['device'] = device
            # Load processor from a base SD model
            from transformers import AutoImageProcessor
            image_processor = AutoImageProcessor.from_pretrained(
                models[embedding_model_name]['PREPROCESSOR_MODEL_ID'],
                # subfolder="feature_extractor",
                revision=None,
            )
            try:
                image_processor = AutoImageProcessor.from_pretrained(
                    models[embedding_model_name]['PREPROCESSOR_MODEL_ID'],
                    subfolder="feature_extractor",
                    revision=None,
                    use_fast=True
                )
            except OSError as e:
                print(f"Trying to load image_processor without subfolder for {models[embedding_model_name]['PREPROCESSOR_MODEL_ID']} due to: {e}")
                try:
                    image_processor = AutoImageProcessor.from_pretrained(models[embedding_model_name]['PREPROCESSOR_MODEL_ID'], use_fast=True)
                except Exception as e_inner:
                    print(f"Could not load AutoImageProcessor for {models[embedding_model_name]['PREPROCESSOR_MODEL_ID']}. Error: {e_inner}")
                    print("Please ensure you have a valid model ID that contains a preprocessor_config.json")
                    exit()
            # Load VAE model
            from diffusers import AutoencoderKL
            try:
                embedding_model = AutoencoderKL.from_pretrained(models[embedding_model_name]['VAE_MODEL_ID']).to(device) # This is from diffusers
            except Exception as e:
                print(f"Could not load VAE model {models[embedding_model_name]['VAE_MODEL_ID']}. Error: {e}")
                exit()

        print(f"Successfully loaded embedding model: {embedding_model_name}")
        return image_processor, embedding_model, device
    except Exception as e:
        print(f"Error loading embedding model: {e}")
        return (None, None, None)

def get_embedding(image_processor, embedding_model, cropped_obb_cv, device, model_name_hint):
    try:
        with torch.no_grad():
            if model_name_hint == 'vit_google':
                cropped_obb_pil = Image.fromarray(cv2.cvtColor(cropped_obb_cv, cv2.COLOR_BGR2RGB))
                inputs = image_processor(images=cropped_obb_pil, return_tensors="pt")
                inputs = {k: v.to(device) for k, v in inputs.items() if isinstance(v, torch.Tensor)}
                outputs = embedding_model(**inputs)
                if hasattr(outputs, 'pooler_output') and outputs.pooler_output is not None:
                    embedding = outputs.pooler_output.squeeze(0).cpu().numpy().tolist()
                elif hasattr(outputs, 'last_hidden_state') and outputs.last_hidden_state is not None:
                    embedding = outputs.last_hidden_state[:, 0, :].squeeze(0).cpu().numpy().tolist()
                else:
                    print(f"Warning: Could not extract a standard embedding from ViT model output.")
                    embedding = []
            
            elif model_name_hint == 'dino':
                cropped_obb_pil = Image.fromarray(cv2.cvtColor(cropped_obb_cv, cv2.COLOR_BGR2RGB))
                inputs = image_processor(images=cropped_obb_pil, return_tensors="pt")
                inputs = {k: v.to(device) for k, v in inputs.items() if isinstance(v, torch.Tensor)}
                outputs = embedding_model(**inputs)
                embedding = outputs.pooler_output.squeeze(0).cpu().numpy().tolist()

            elif model_name_hint == 'dino_giant':
                cropped_obb_pil = Image.fromarray(cv2.cvtColor(cropped_obb_cv, cv2.COLOR_BGR2RGB))
                inputs = image_processor(images=cropped_obb_pil, return_tensors="pt")
                inputs = {k: v.to(device) for k, v in inputs.items() if isinstance(v, torch.Tensor)}
                outputs = embedding_model(**inputs)
                embedding = outputs.pooler_output.squeeze(0).cpu().numpy().tolist()

            elif model_name_hint == 'clip':
                cropped_obb_pil = Image.fromarray(cv2.cvtColor(cropped_obb_cv, cv2.COLOR_BGR2RGB))
                inputs = image_processor(images=cropped_obb_pil, return_tensors="pt")
                inputs = {k: v.to(device) for k, v in inputs.items() if isinstance(v, torch.Tensor)}
                outputs = embedding_model(**inputs)
                # CLIPVisionModel returns pooler_output [1, 768] — L2-normalize for cosine similarity
                feats = outputs.pooler_output
                feats = feats / feats.norm(p=2, dim=-1, keepdim=True)
                embedding = feats.squeeze(0).cpu().numpy().tolist()

            elif model_name_hint == 'siglip':
                cropped_obb_pil = Image.fromarray(cv2.cvtColor(cropped_obb_cv, cv2.COLOR_BGR2RGB))
                inputs = image_processor(images=cropped_obb_pil, return_tensors="pt")
                inputs = {k: v.to(device) for k, v in inputs.items() if isinstance(v, torch.Tensor)}
                outputs = embedding_model(**inputs)
                # SiglipVisionModel returns pooler_output [1, 1152]
                feats = outputs.pooler_output
                feats = feats / feats.norm(p=2, dim=-1, keepdim=True)
                embedding = feats.squeeze(0).cpu().numpy().tolist()

            elif model_name_hint == 'radio':
                cropped_obb_pil = Image.fromarray(cv2.cvtColor(cropped_obb_cv, cv2.COLOR_BGR2RGB))
                # image_processor is the timm transform: PIL -> Tensor [3, H, W]
                tensor = image_processor(cropped_obb_pil).unsqueeze(0).to(device)
                outputs = embedding_model(tensor)
                # RADIO returns (summary, spatial_features); summary is [1, D]
                summary = outputs[0] if isinstance(outputs, (tuple, list)) else outputs
                embedding = summary.squeeze(0).cpu().numpy().tolist()

            elif model_name_hint == 'sd-vae':
                cropped_obb_pil = Image.fromarray(cv2.cvtColor(cropped_obb_cv, cv2.COLOR_BGR2RGB))
                # padded_abb_cv = pad_to_square(cropped_obb_pil)
                image = models[model_name_hint]['transform'](cropped_obb_pil)
                processed_image = image_processor(image, return_tensors="pt").to(device).pixel_values
                encoder_output = embedding_model.encode(processed_image)
                latent_dist = encoder_output.latent_dist
                latents = latent_dist.sample()
                if hasattr(embedding_model.config, 'scaling_factor') and embedding_model.config.scaling_factor is not None:
                    latents = latents * embedding_model.config.scaling_factor
                embedding = latents.flatten().cpu().numpy().tolist()
            
            else:
                # Generic fallback for other models, or handle as an error
                print(f"Warning: Untested or unsupported model_name_hint '{model_name_hint}' for embedding extraction. Attempting generic method.")
                outputs = embedding_model(**inputs) # Try generic call
                if hasattr(outputs, 'pooler_output') and outputs.pooler_output is not None:
                    embedding = outputs.pooler_output.squeeze(0).cpu().numpy().tolist()
                elif hasattr(outputs, 'last_hidden_state') and outputs.last_hidden_state is not None:
                    embedding = outputs.last_hidden_state[:, 0, :].squeeze(0).cpu().numpy().tolist() # CLS token
                else:
                    print(f"Warning: Could not extract a standard embedding from model output for {model_name_hint}.")
                    embedding = []
        return embedding

    except Exception as e_embed:
        print(f"Error during embedding extraction for an object (model: {model_name_hint}): {e_embed}")
        import traceback
        traceback.print_exc() # Print full traceback for debugging
        return [] # Return empty list on error
