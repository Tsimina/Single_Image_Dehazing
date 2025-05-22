import os
import argparse
from PIL import Image


def resize_image(input_path, output_path, size=(256, 256)):
    try:
        print(f"Opening image: {input_path}")
        with Image.open(input_path) as img:
            img = img.convert('RGB')  # Ensure consistent format
            img_resized = img.resize(size, resample=Image.LANCZOS)
            img_resized.save(output_path)
            print(f"Resized and saved: {output_path}")
    except Exception as e:
        print(f"Error processing {input_path}: {e}")


def process_directory(input_dir, output_dir, size=(256, 256), extensions=None):
    if not os.path.exists(input_dir):
        print(f"Input directory does not exist: {input_dir}")
        return

    if extensions is None:
        extensions = ['.jpg', '.jpeg', '.png', '.bmp', '.gif', '.tiff']

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
                resize_image(input_path, output_path, size)
            else:
                print(f"Skipping non-image file: {filename}")


def main():
    parser = argparse.ArgumentParser(description="Resize all images in a directory to a given size, preserving original names.")
    parser.add_argument('input_dir', help='Path to the input directory containing images')
    parser.add_argument('output_dir', help='Path to the output directory for resized images')
    parser.add_argument('--width', type=int, default=256, help='Target width in pixels')
    parser.add_argument('--height', type=int, default=256, help='Target height in pixels')
    args = parser.parse_args()

    size = (args.width, args.height)
    print(f"Target size: {size[0]}x{size[1]}")
    process_directory(args.input_dir, args.output_dir, size)


if __name__ == '__main__':
    main()
