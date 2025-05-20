import os
import torch
from torchvision import transforms
from torchvision.utils import save_image
from PIL import Image
import glob
from model.dcp_unet import DehazingUNet as DehazingModel


def run_inference(model_path, input_dir, output_dir, device):
    os.makedirs(output_dir, exist_ok=True)

    transform = transforms.Compose([
        transforms.Resize((256, 256)),
        transforms.ToTensor()
    ])

    model = DehazingModel().to(device)
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.eval()

    image_paths = glob.glob(os.path.join(input_dir, "*.jpg"))
    with torch.no_grad():
        for path in image_paths:
            image = Image.open(path).convert("RGB")
            input_tensor = transform(image).unsqueeze(0).to(device)
            output = model(input_tensor)
            filename = os.path.basename(path)
            save_image(output, os.path.join(output_dir, f"dehazed_{filename}"))
            print(f"Saved: dehazed_{filename}")


if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    run_inference(
        model_path="model.pth",
        input_dir="images",      
        output_dir="outputs",    
        device=device
    )