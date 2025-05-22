import os
import sys
# Add parent directories to sys.path for module imports
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '.')))
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
import torch
from torchvision import transforms
from torchvision.utils import save_image
from PIL import Image
import glob
import argparse
import time

def get_model(model_type):
   # Returns the appropriate model based on the model_type argument.
    if model_type == "unet":
        from model_configurations.dcp_unet import DehazingUNet as DehazingModel
        return DehazingModel()
    elif model_type == "sar_unet":
        from model_configurations.dcp_sar_unet import DehazingUNet as DehazingModel
        return DehazingModel(use_sar=True)
    else:
        raise ValueError("Unknown model_type. Use 'unet' or 'sar_unet'.")

def run_inference(model_path, input_dir, output_dir, device, model_type, sar_dir=None, sar_ext=".png"):
    """
    Runs inference on all images in input_dir using the specified model.
    If model_type is 'sar_unet', matches each hazy image with its corresponding SAR image.
    """
    os.makedirs(output_dir, exist_ok=True)

    # Define image transformation pipeline
    transform = transforms.Compose([
        transforms.Resize((256, 256)),
        transforms.ToTensor()
    ])

    # Load model and weights
    model = get_model(model_type).to(device)
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.eval()

    # Get all .jpg images from input_dir
    image_paths = glob.glob(os.path.join(input_dir, "*.jpg"))
    with torch.no_grad():
        for path in image_paths:
            image = Image.open(path).convert("RGB")
            input_tensor = transform(image).unsqueeze(0).to(device)
            start_time = time.time()
            if model_type == "sar_unet":
                # Match SAR image by name and extension
                base = os.path.splitext(os.path.basename(path))[0]
                sar_folder = sar_dir if sar_dir else input_dir
                sar_path = os.path.join(sar_folder, f"{base}_sar{sar_ext}")
                if not os.path.exists(sar_path):
                    print(f"Warning: SAR image not found for {path}, skipping.")
                    continue
                sar = Image.open(sar_path).convert("L")
                sar_tensor = transform(sar).unsqueeze(0).to(device)
                output = model(input_tensor, sar_tensor)
            else:
                output = model(input_tensor)
            elapsed = time.time() - start_time
            filename = os.path.basename(path)
            save_image(output, os.path.join(output_dir, f"dehazed_{filename}"))
            print(f"Saved: dehazed_{filename} | Inference time: {elapsed:.4f} seconds")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    # Model and directory arguments
    parser.add_argument("--model", type=str, required=True, help="Path to model .pth file")
    parser.add_argument("--input_dir", type=str, required=True, help="Directory with input images")
    parser.add_argument("--output_dir", type=str, required=True, help="Directory to save dehazed images")
    parser.add_argument("--device", type=str, default="cuda", help="cuda or cpu")
    parser.add_argument("--model_type", type=str, default="unet", help="Model type: 'unet' or 'sar_unet'")
    parser.add_argument("--sar_dir", type=str, help="Directory with SAR images (default: same as input_dir)")
    parser.add_argument("--sar_ext", type=str, default=".png", help="SAR image extension (default: .png)")
    args = parser.parse_args()

    # Select device
    device = torch.device(args.device if torch.cuda.is_available() and args.device == "cuda" else "cpu")

    # Run inference
    run_inference(
        model_path=args.model,
        input_dir=args.input_dir,
        output_dir=args.output_dir,
        device=device,
        model_type=args.model_type,
        sar_dir=args.sar_dir,
        sar_ext=args.sar_ext
    )