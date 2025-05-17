import deeplake
import os
from PIL import Image
import numpy as np
from tqdm import tqdm

# === Setări de salvare ===
save_dir = "reside_download"
os.makedirs(f"{save_dir}/hazy", exist_ok=True)
os.makedirs(f"{save_dir}/clear", exist_ok=True)

# === Încarcă datasetul din hub Activeloop ===
ds = deeplake.load("hub://activeloop/reside")

# === Inițializează indecși pentru denumiri de fișiere ===
hazy_count = 0
clear_count = 0

# === Iterare pe obiectul dataset, NU pe range(len(ds)) ===
for sample in tqdm(ds, desc="Salvez imagini"):
    image = sample['images'].numpy()
    label = sample['labels'].numpy().item()  # eticheta: 'hazy' sau 'clear'

    if label == "hazy":
        filename = f"haze_{hazy_count:04}.jpg"
        Image.fromarray(image).save(os.path.join(save_dir, "hazy", filename))
        hazy_count += 1

    elif label == "clear":
        filename = f"gt_{clear_count:04}.jpg"
        Image.fromarray(image).save(os.path.join(save_dir, "clear", filename))
        clear_count += 1

print(f"Saved in '{save_dir}'")
