import os
import argparse
from PIL import Image, ImageFilter
import numpy as np

def fake_sar_from_rgb(rgb_img):
    sar = rgb_img.convert("L")
    sar = sar.filter(ImageFilter.FIND_EDGES)
    sar_np = np.array(sar).astype(np.float32) / 255.0
    noise = np.random.normal(0, 0.05, sar_np.shape)
    sar_np = np.clip(sar_np + noise, 0, 1)
    sar = Image.fromarray((sar_np * 255).astype(np.uint8))
    return sar

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--input_dir", type=str, required=True, help="Folder cu imagini RGB")
    parser.add_argument("--output_dir", type=str, required=True, help="Folder unde se salvează imaginile SAR")
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)

    for fname in os.listdir(args.input_dir):
        if fname.lower().endswith(('.jpg', '.jpeg', '.png')):
            img = Image.open(os.path.join(args.input_dir, fname)).convert("RGB")
            sar = fake_sar_from_rgb(img)
            out_name = fname.rsplit('.', 1)[0] + '.png'
            sar.save(os.path.join(args.output_dir, out_name))
    print("SAR images generated in", args.output_dir)