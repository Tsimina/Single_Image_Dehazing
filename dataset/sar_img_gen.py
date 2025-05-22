import os
import argparse
from PIL import Image, ImageFilter
import numpy as np

"""
Generate a synthetic SAR-like image from an RGB image by converting to grayscale and adding noise.
"""
def generate_sar_image(input_path, output_path, noise_level=25):
    try:
        print(f"Opening image: {input_path}")
        with Image.open(input_path) as img:
            # Convert to grayscale to simulate SAR
            sar_img = img.convert('L')
            # Convert to numpy array for noise addition
            sar_np = np.array(sar_img).astype(np.float32)
            # Add Gaussian noise
            noise = np.random.normal(0, noise_level, sar_np.shape)
            sar_np_noisy = sar_np + noise
            sar_np_noisy = np.clip(sar_np_noisy, 0, 255).astype(np.uint8)
            # Convert back to PIL Image
            sar_img_noisy = Image.fromarray(sar_np_noisy)
            # Optionally apply a filter to simulate SAR texture
            sar_img_noisy = sar_img_noisy.filter(ImageFilter.GaussianBlur(radius=1))
            sar_img_noisy.save(output_path)
            print(f"SAR image saved: {output_path}")
    except Exception as e:
        print(f"Error processing {input_path}: {e}")

"""
Process all images in a directory, generating SAR images for each.
"""
def process_directory(input_dir, output_dir, noise_level=25, extensions=None):
    if not os.path.exists(input_dir):
        print(f"Input directory does not exist: {input_dir}")
        return

    if extensions is None:
        extensions = ['.jpg', '.jpeg', '.png', '.bmp', '.tiff']

    os.makedirs(output_dir, exist_ok=True)
    print(f"Processing directory: {input_dir}")

    for root, _, files in os.walk(input_dir):
        rel_path = os.path.relpath(root, input_dir)
        target_dir = os.path.join(output_dir, rel_path)
        os.makedirs(target_dir, exist_ok=True)
        print(f"Entering folder: {root}, found {len(files)} files")

        for filename in files:
            name, ext = os.path.splitext(filename)
            if ext.lower() in extensions:
                input_path = os.path.join(root, filename)
                output_path = os.path.join(target_dir, filename)
                generate_sar_image(input_path, output_path, noise_level)
            else:
                print(f"Skipping non-image file: {filename}")

"""
Main function to parse arguments and start SAR image generation.
"""
def main():
    parser = argparse.ArgumentParser(description="Generate synthetic SAR images from RGB images in a directory.")
    parser.add_argument('input_dir', help='Path to the input directory containing images')
    parser.add_argument('output_dir', help='Path to the output directory for SAR images')
    parser.add_argument('--noise', type=float, default=25, help='Noise level for SAR simulation')
    args = parser.parse_args()

    process_directory(args.input_dir, args.output_dir, noise_level=args.noise)

if __name__ == '__main__':
    main()