import os
from PIL import Image, ImageFilter
import numpy as np

input_dir = "test_application"
output_dir = "test_application"
os.makedirs(output_dir, exist_ok=True)

def fake_sar_from_rgb(rgb_img):
    sar = rgb_img.convert("L")
    sar = sar.filter(ImageFilter.FIND_EDGES)
    sar_np = np.array(sar).astype(np.float32) / 255.0
    noise = np.random.normal(0, 0.05, sar_np.shape)
    sar_np = np.clip(sar_np + noise, 0, 1)
    sar = Image.fromarray((sar_np * 255).astype(np.uint8))
    return sar

for fname in os.listdir(input_dir):
    if fname.lower().endswith(('.jpg', '.jpeg', '.png')):
        img = Image.open(os.path.join(input_dir, fname)).convert("RGB")
        sar = fake_sar_from_rgb(img)
        # Salvează cu extensia .png (sau păstrează extensia originală dacă vrei)
        out_name = fname.rsplit('.', 1)[0] + '.png'
        sar.save(os.path.join(output_dir, out_name))
print("SAR images generated in", output_dir)