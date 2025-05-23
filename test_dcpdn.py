# import os
# import torch
# import matplotlib.pyplot as plt
# from torchvision import transforms
# from PIL import Image
# from models.dcpdn import DCPDN
# from skimage.metrics import peak_signal_noise_ratio as psnr
# from skimage.metrics import structural_similarity as ssim


# # Configuration
# ROOT = "D:\Ari"
# MODEL_PATH = "dcpdn_35_scheduler_epochs.pth"  # Path to your trained model
# TEST_IMAGE_DIR = "test_hazy"            # Folder containing hazy test images
# OUTPUT_DIR = "test_results"                 # Where to save outputs
# os.makedirs(OUTPUT_DIR, exist_ok=True)

# # Device setup
# device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# # Load model
# model = DCPDN().to(device)
# model.load_state_dict(torch.load(MODEL_PATH, map_location=device))
# model.eval()

# # Image preprocessing
# transform = transforms.Compose([
#     transforms.Resize((256, 256)),
#     transforms.ToTensor(),
# ])

# # Process each test image
# for img_name in os.listdir(TEST_IMAGE_DIR)[:5]:  # Process first 5 images
#     # Load hazy image
#     hazy_path = os.path.join(TEST_IMAGE_DIR, img_name)
#     hazy_img = Image.open(hazy_path).convert('RGB')
#     hazy_tensor = transform(hazy_img).unsqueeze(0).to(device)
    
#     # Get clear image path (assuming naming convention: 0025_0.8_0.1.png -> 0025.png)
#     clear_id = img_name.split('_')[0]
#     clear_path = os.path.join(os.path.dirname(TEST_IMAGE_DIR), "clear", f"{clear_id}.png")
    
#     # Run model
#     with torch.no_grad():
#         dehazed_tensor, trans_map, _ = model(hazy_tensor)
    
#     # Convert tensors to images
#     dehazed_img = transforms.ToPILImage()(dehazed_tensor.squeeze().cpu())
#     trans_map_img = transforms.ToPILImage()(trans_map.squeeze().cpu())
#     trans_map_img.save(os.path.join(OUTPUT_DIR, f"trans_map_{img_name}"))

    
#     # Load ground truth clear image if available
#     clear_img = None
#     if os.path.exists(clear_path):
#         clear_img = Image.open(clear_path).convert('RGB')
#         clear_img = clear_img.resize((256, 256))
    
#     # Create comparison plot
#     plt.figure(figsize=(15, 5))
    
#     plt.subplot(1, 4, 1)
#     plt.imshow(hazy_img)
#     plt.title("Hazy Input")
#     plt.axis('off')
    
#     plt.subplot(1, 4, 2)
#     plt.imshow(dehazed_img)
#     plt.title("Dehazed Output")
#     plt.axis('off')
    
#     plt.subplot(1, 4, 3)
#     plt.imshow(trans_map_img, cmap='gray')
#     plt.title("Depth Map")
#     plt.axis('off')
    
#     if clear_img:
#         clear_tensor = transform(clear_img).unsqueeze(0).to(device)
#         test_psnr = psnr(clear_tensor.cpu().numpy(),  # Convert to numpy array
#                     dehazed_tensor.cpu().numpy(), 
#                     data_range=1.0)
#         test_ssim = ssim(clear_tensor.cpu().numpy()[0].transpose(1,2,0),
#                 dehazed_tensor.cpu().numpy()[0].transpose(1,2,0),
#                 channel_axis=-1, data_range=1.0)
#         print(f"PSNR: {test_psnr:.2f} dB | SSIM: {test_ssim:.4f}")
#         plt.subplot(1, 4, 4)
#         plt.imshow(clear_img)
#         plt.title("Ground Truth")
#         plt.axis('off')
    
#     # Save and show results
#     output_path = os.path.join(OUTPUT_DIR, f"result_{img_name}")
#     plt.savefig(output_path, bbox_inches='tight')
#     plt.show()
#     plt.close()
    
#     print(f"Saved results for {img_name} to {output_path}")


import os
import torch
import matplotlib.pyplot as plt
from torchvision import transforms
from PIL import Image
from models.dcpdn import DCPDN
from skimage.metrics import peak_signal_noise_ratio as psnr
from skimage.metrics import structural_similarity as ssim

# Configuration
ROOT = "S:\master_poli\sem_2\iarnp\proiect\final"
MODEL_PATH = "dcpdn_60_scheduler_epochs.pth"  # Path to your trained model
TEST_IMAGE_DIR = "test_natural_image.jpeg"            # Folder containing hazy test images
OUTPUT_DIR = "test_results"                 # Where to save outputs
os.makedirs(OUTPUT_DIR, exist_ok=True)

# Device setup
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Load model
model = DCPDN().to(device)
model.load_state_dict(torch.load(MODEL_PATH, map_location=device))
model.eval()

# Image preprocessing
transform = transforms.Compose([
    transforms.Resize((256, 256)),
    transforms.ToTensor(),
])

# Process each test image
for img_name in os.listdir(TEST_IMAGE_DIR)[:5]:  # Process first 5 images
    # Load hazy image
    hazy_path = os.path.join(TEST_IMAGE_DIR, img_name)
    hazy_img = Image.open(hazy_path).convert('RGB')
    hazy_tensor = transform(hazy_img).unsqueeze(0).to(device)
    
    # Create resized version for display
    resized_img = hazy_img.resize((256, 256))
    
    # Get clear image path (assuming naming convention: 0025_0.8_0.1.png -> 0025.png)
    clear_id = img_name.split('_')[0]
    clear_path = os.path.join(os.path.dirname(TEST_IMAGE_DIR), "clear", f"{clear_id}.png")
    
    # Run model
    with torch.no_grad():
        dehazed_tensor, trans_map, _ = model(hazy_tensor)
    
    # Convert tensors to images
    dehazed_img = transforms.ToPILImage()(dehazed_tensor.squeeze().cpu())
    trans_map_img = transforms.ToPILImage()(trans_map.squeeze().cpu())
    trans_map_img.save(os.path.join(OUTPUT_DIR, f"trans_map_60_epochs{img_name}"))

    # Load ground truth clear image if available
    clear_img = None
    if os.path.exists(clear_path):
        clear_img = Image.open(clear_path).convert('RGB')
        clear_img = clear_img.resize((256, 256))
    
    # Create comparison plot - now with 5 subplots
    plt.figure(figsize=(20, 5))
    
    plt.subplot(1, 5, 1)
    plt.imshow(hazy_img)
    plt.title("Original Hazy Input")
    plt.axis('off')
    
    plt.subplot(1, 5, 2)
    plt.imshow(resized_img)
    plt.title("Resized Input (256x256)")
    plt.axis('off')
    
    plt.subplot(1, 5, 3)
    plt.imshow(dehazed_img)
    plt.title("Dehazed Output")
    plt.axis('off')
    
    plt.subplot(1, 5, 4)
    plt.imshow(trans_map_img, cmap='gray')
    plt.title("Depth Map")
    plt.axis('off')
    
    if clear_img:
        clear_tensor = transform(clear_img).unsqueeze(0).to(device)
        test_psnr = psnr(clear_tensor.cpu().numpy(),  # Convert to numpy array
                    dehazed_tensor.cpu().numpy(), 
                    data_range=1.0)
        test_ssim = ssim(clear_tensor.cpu().numpy()[0].transpose(1,2,0),
                dehazed_tensor.cpu().numpy()[0].transpose(1,2,0),
                channel_axis=-1, data_range=1.0)
        print(f"PSNR: {test_psnr:.2f} dB | SSIM: {test_ssim:.4f}")
        plt.subplot(1, 5, 5)
        plt.imshow(clear_img)
        plt.title("Ground Truth")
        plt.axis('off')

    # Save and show results
    output_path = os.path.join(OUTPUT_DIR, f"result_60_epochs{img_name}")
    plt.savefig(output_path, bbox_inches='tight', dpi=100)
    plt.show()
    plt.close()
    
    print(f"Saved results for {img_name} to {output_path}_60_epochs")