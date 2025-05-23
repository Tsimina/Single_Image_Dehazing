import os
import torch
import matplotlib.pyplot as plt
from torchvision import transforms
from PIL import Image
from models_ia2.dcpdn_midas import DCPDN_MiDaS
import numpy as np

# Config
MODEL_PATH = "saved_models/dcpdn_midas_best.pth"
TEST_IMAGE_DIR = "hazy"
OUTPUT_DIR = "test_results"
os.makedirs(OUTPUT_DIR, exist_ok=True)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Load model
model = DCPDN_MiDaS().to(device)
model.load_state_dict(torch.load(MODEL_PATH, map_location=device))
model.eval()

# Transforms
transform = transforms.Compose([
    transforms.Resize((384, 384)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

# Visualization function
def denormalize(tensor):
    """Convert normalized tensor back to PIL image"""
    mean = torch.tensor([0.485, 0.456, 0.406]).view(1,3,1,1).to(device)
    std = torch.tensor([0.229, 0.224, 0.225]).view(1,3,1,1).to(device)
    tensor = tensor * std + mean
    tensor = torch.clamp(tensor, 0, 1)
    return transforms.ToPILImage()(tensor.squeeze().cpu())

for img_name in os.listdir(TEST_IMAGE_DIR)[:5]:  # Process first 5 images
    # Load original hazy image
    hazy_path = os.path.join(TEST_IMAGE_DIR, img_name)
    hazy_img = Image.open(hazy_path).convert('RGB')
    
    # Create resized version (exactly what model sees)
    resized_img = hazy_img.resize((384, 384))
    hazy_tensor = transform(hazy_img).unsqueeze(0).to(device)
    
    # Get ground truth path
    clear_id = img_name.split('_')[0]
    clear_path = os.path.join(os.path.dirname(TEST_IMAGE_DIR), "clear", f"{clear_id}.png")
    
    # Run model
    with torch.no_grad():
        dehazed_tensor, trans_map, _ = model(hazy_tensor)
    
    # Convert outputs
    dehazed_img = denormalize(dehazed_tensor)
    trans_map_img = transforms.ToPILImage()(trans_map.squeeze().cpu())
    
    # Create comparison plot with 5 columns
    plt.figure(figsize=(22, 5))
    
    # Column 1: Original hazy image
    plt.subplot(1, 5, 1)
    plt.imshow(hazy_img)
    plt.title("Original Hazy Image", fontsize=10)
    plt.axis('off')
    
    # Column 2: Resized input (model's actual input)
    plt.subplot(1, 5, 2)
    plt.imshow(resized_img)
    plt.title("Resized Input (384x384)", fontsize=10)
    plt.axis('off')
    
    # Column 3: Dehazed result
    plt.subplot(1, 5, 3)
    plt.imshow(dehazed_img)
    plt.title("Dehazed Output", fontsize=10)
    plt.axis('off')
    
    # Column 4: Depth map
    plt.subplot(1, 5, 4)
    plt.imshow(trans_map_img, cmap='magma')
    plt.colorbar(fraction=0.046, pad=0.04)
    plt.title("MiDaS Depth Map", fontsize=10)
    plt.axis('off')
    
    # Column 5: Ground truth (if available)
    if os.path.exists(clear_path):
        clear_img = Image.open(clear_path).convert('RGB').resize((384, 384))
        plt.subplot(1, 5, 5)
        plt.imshow(clear_img)
        plt.title("Ground Truth", fontsize=10)
        plt.axis('off')
    else:
        plt.subplot(1, 5, 5)
        plt.text(0.5, 0.5, 'No Ground Truth', ha='center', va='center')
        plt.axis('off')
    
    # Save and display
    plt.tight_layout()
    output_path = os.path.join(OUTPUT_DIR, f"result_{os.path.splitext(img_name)[0]}.png")
    plt.savefig(output_path, bbox_inches='tight', dpi=120)
    plt.close()
    
    print(f"Saved visualization for {img_name} to {output_path}_dcpdn_midas")