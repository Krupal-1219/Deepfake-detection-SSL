import torch

print("Loading DINOv2 Large with Registers (ViT-L/14)...")

# 1. Load the model
model = torch.hub.load('facebookresearch/dinov2', 'dinov2_vitl14_reg')

# 2. Move to GPU
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model.to(device)

# 3. SET TO EVAL AND FREEZE (Crucial for the "Online" Frozen way)
model.eval()
for param in model.parameters():
    param.requires_grad = False

print(f"Model successfully loaded and frozen on: {device}")

# --- Example of how to use it in your pipeline ---

def get_dinov2_features(batch_tensor):
    """
    batch_tensor: [Batch, 3, 224, 224]
    """
    with torch.no_grad(): # Saves a massive amount of VRAM during training
        # .forward_features() returns a dict containing everything you need
        outputs = model.forward_features(batch_tensor)
        
        # 1. Global face identity (1024-d)
        cls_token = outputs['x_norm_clstoken'] 
        
        # 2. Spatial patch tokens (256 tokens, each 1024-d)
        # These are used for your "Main Contribution" (FGW distance)
        patch_tokens = outputs['x_norm_patchtokens']
        
    return cls_token, patch_tokens