# visualize_single_image.py
import torch
from torchvision import transforms
from torchvision.utils import make_grid
import matplotlib.pyplot as plt
from PIL import Image
from model.dcp_unet import DehazingUNet as DehazingModel


def visualize_dehazed_image(image_path, model_path, device):
    transform = transforms.Compose([
        transforms.Resize((256, 256)),
        transforms.ToTensor()
    ])

    image = Image.open(image_path).convert("RGB")
    input_tensor = transform(image).unsqueeze(0).to(device)

    model = DehazingModel().to(device)
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.eval()

    with torch.no_grad():
        output = model(input_tensor)

    # Pregătește pentru afișare
    images_to_show = torch.cat([input_tensor, output], dim=0).cpu()
    grid = make_grid(images_to_show, nrow=2)

    plt.figure(figsize=(10, 5))
    plt.title("Left: Hazy | Right: Dehazed")
    plt.imshow(grid.permute(1, 2, 0))
    plt.axis("off")
    plt.show()


if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    visualize_dehazed_image(
        image_path="test_application\haze_img2.jpg",
        model_path="test_application\model.pth",
        device=device
    )
