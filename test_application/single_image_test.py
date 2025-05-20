import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import argparse
import torch
from torchvision import transforms
from torchvision.utils import make_grid
import matplotlib.pyplot as plt
from PIL import Image
from model.dcp_sar_unet import DehazingUNet as DehazingModel

# Function to visualize the dehazed image
def visualize_dehazed_image(image_path, model_path, device, use_sar=False, sar_path=None):
    transform_rgb = transforms.Compose([
        transforms.Resize((256, 256)),
        transforms.ToTensor()
    ])
    image = Image.open(image_path).convert("RGB")
    input_tensor = transform_rgb(image).unsqueeze(0).to(device)

    if use_sar:
        if sar_path is None:
            raise ValueError("Trebuie să specifici --sar_path pentru modelul SAR!")
        transform_sar = transforms.Compose([
            transforms.Resize((256, 256)),
            transforms.ToTensor()
        ])
        sar = Image.open(sar_path).convert("L")
        sar_tensor = transform_sar(sar).unsqueeze(0).to(device)
        model = DehazingModel(use_sar=True).to(device)
        model.load_state_dict(torch.load(model_path, map_location=device))
        model.eval()
        with torch.no_grad():
            output = model(input_tensor, sar_tensor)
    else:
        model = DehazingModel(use_sar=False).to(device)
        model.load_state_dict(torch.load(model_path, map_location=device))
        model.eval()
        with torch.no_grad():
            output = model(input_tensor)

    images_to_show = torch.cat([input_tensor, output], dim=0).cpu()
    grid = make_grid(images_to_show, nrow=2)
    plt.figure(figsize=(10, 5))
    plt.title("Left: Hazy | Right: Dehazed")
    plt.imshow(grid.permute(1, 2, 0))
    plt.axis("off")
    plt.show()

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--image", type=str, required=True, help="Calea către imaginea hazy")
    parser.add_argument("--model", type=str, required=True, help="Calea către modelul salvat")
    parser.add_argument("--use_sar", action="store_true", help="Folosește modelul cu SAR")
    parser.add_argument("--sar_path", type=str, help="Calea către imaginea SAR (dacă folosești SAR)")
    parser.add_argument("--device", type=str, default="cuda", help="cuda sau cpu")
    args = parser.parse_args()

# Block for running the script
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    visualize_dehazed_image(
        image_path=args.image,
        model_path=args.model,
        device=device,
        use_sar=args.use_sar,
        sar_path=args.sar_path
    )