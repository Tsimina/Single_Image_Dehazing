
import os
import time
import torch
import matplotlib.pyplot as plt
from torchvision import transforms
from PIL import Image
from models_ia2.aod_net import AODNet
# from models_ia2.dcpdn import DCPDN
from skimage.metrics import peak_signal_noise_ratio as psnr
from skimage.metrics import structural_similarity as ssim

# Configuration
ROOT = r"D:\Ari"
MODEL_PATH = os.path.join(ROOT, "dcpdn_60_scheduler_epochs.pth")
TEST_IMAGE_DIR = os.path.join(ROOT, "test_hazy")
OUTPUT_DIR = os.path.join(ROOT, "test_results")
os.makedirs(OUTPUT_DIR, exist_ok=True)

# Device setup
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Load model
model = AODNet().to(device)
# model = DCPDN().to(device)
model.load_state_dict(torch.load(MODEL_PATH, map_location=device))
model.eval()

# Image preprocessing
transform = transforms.Compose([
    transforms.Resize((256, 256)),
    transforms.ToTensor(),
])

# Timing storage
times = []

# Process each test image
for img_name in sorted(os.listdir(TEST_IMAGE_DIR))[:5]:
    hazy_path = os.path.join(TEST_IMAGE_DIR, img_name)
    hazy_img = Image.open(hazy_path).convert('RGB')
    hazy_tensor = transform(hazy_img).unsqueeze(0).to(device)

    resized_img = hazy_img.resize((256, 256))

    # Ground truth path
    clear_id = img_name.split('_')[0]
    clear_path = os.path.join(ROOT, "clear", f"{clear_id}.png")

    # Inference + timing
    start = time.time()
    with torch.no_grad():
        dehazed_tensor, trans_map, _ = model(hazy_tensor)
    end = time.time()
    elapsed = end - start
    times.append(elapsed)
    print(f"{img_name} inference time: {elapsed*1000:.1f} ms")

    # Convert outputs to images
    dehazed_img = transforms.ToPILImage()(dehazed_tensor.squeeze().cpu())
    trans_map_img = transforms.ToPILImage()(trans_map.squeeze().cpu())
    trans_map_img.save(os.path.join(OUTPUT_DIR, f"trans_map_{img_name}"))

    # Load and resize ground truth if available
    clear_img = None
    if os.path.exists(clear_path):
        clear_img = Image.open(clear_path).convert('RGB').resize((256,256))

    # Plot
    plt.figure(figsize=(20,5))
    plt.subplot(1,5,1); plt.imshow(hazy_img); plt.title("Original Hazy"); plt.axis('off')
    plt.subplot(1,5,2); plt.imshow(resized_img); plt.title("Resized Input"); plt.axis('off')
    plt.subplot(1,5,3); plt.imshow(dehazed_img); plt.title("Dehazed Output"); plt.axis('off')
    plt.subplot(1,5,4); plt.imshow(trans_map_img, cmap='gray'); plt.title("Trans. Map"); plt.axis('off')

    if clear_img:
        clear_tensor = transform(clear_img).unsqueeze(0).to(device)
        test_psnr = psnr(clear_tensor.cpu().numpy(), dehazed_tensor.cpu().numpy(), data_range=1.0)
        test_ssim = ssim(
            clear_tensor.cpu().numpy()[0].transpose(1,2,0),
            dehazed_tensor.cpu().numpy()[0].transpose(1,2,0),
            channel_axis=-1, data_range=1.0
        )
        print(f"PSNR: {test_psnr:.2f} dB | SSIM: {test_ssim:.4f}")
        plt.subplot(1,5,5); plt.imshow(clear_img); plt.title("Ground Truth"); plt.axis('off')

    out_path = os.path.join(OUTPUT_DIR, f"result_{img_name}")
    plt.savefig(out_path, bbox_inches='tight', dpi=100)
    plt.show()
    plt.close()
    print(f"Saved {out_path}")

# Summary timing
avg_time = sum(times) / len(times)
print(f"\nAverage inference time per image: {avg_time*1000:.1f} ms")
