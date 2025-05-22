import os
from PIL import Image
from torch.utils.data import Dataset

"""
Custom dataset for loading triplets of (hazy RGB, SAR, clear RGB) images.
Assumes directory structure:
    clear_rsz/ — ground truth clear images
    hazy_rsz/  — hazy images
    hazy_sar/  — corresponding SAR images
"""
class MultiHazySARDataset(Dataset):
    """
    Each sample consists of:
        hazy: hazy RGB image (e.g. 3068_0.8_0.1.jpg)
        sar:  corresponding SAR image (e.g. 3068_0.8_0.1.png)
        clear: ground truth clear RGB image (e.g. 3068.jpg)
    """
    def __init__(self, root_dir, transform_rgb=None, transform_sar=None, clear_ext="jpg", haze_exts=("jpg", "png"), sar_ext="png"):
        """
        root_dir: Root directory containing the subfolders.
        transform_rgb: Transformations for RGB images.
        transform_sar: Transformations for SAR images.
        clear_ext: Extension for clear images.
        haze_exts: Extensions for hazy images.
        sar_ext: Extension for SAR images.
        """
        clear_dir = os.path.join(root_dir, "clear_rsz")
        haze_dir  = os.path.join(root_dir, "hazy_rsz")
        sar_dir   = os.path.join(root_dir, "hazy_sar")
        # Check that all required directories exist
        assert os.path.isdir(clear_dir) and os.path.isdir(haze_dir) and os.path.isdir(sar_dir), "Structura folderelor invalidă"
        self.transform_rgb = transform_rgb
        self.transform_sar = transform_sar
        self.triples = []
        # Map clear image stems to their paths
        clear_map = {}
        for cp in os.listdir(clear_dir):
            if cp.endswith(f".{clear_ext}"):
                stem = os.path.splitext(cp)[0]
                clear_map[stem] = os.path.join(clear_dir, cp)
        # For each hazy image, find matching clear and SAR images
        for hp in os.listdir(haze_dir):
            if any(hp.endswith(f".{ext}") for ext in haze_exts):
                stem = os.path.basename(hp).split('_')[0]
                if stem in clear_map:
                    sar_name = os.path.splitext(hp)[0] + f".{sar_ext}"
                    sar_path = os.path.join(sar_dir, sar_name)
                    if os.path.exists(sar_path):
                        self.triples.append((os.path.join(haze_dir, hp), sar_path, clear_map[stem]))
        # Ensure at least one valid triplet exists
        assert self.triples, "No valid pair (hazy, sar, clear)"

        # Print the first 5 triplets for verification
        for i, (hazy_path, sar_path, clear_path) in enumerate(self.triples[:5]):
            print(f"{i+1}: Hazy: {os.path.basename(hazy_path)} | SAR: {os.path.basename(sar_path)} | Clear: {os.path.basename(clear_path)}")

    def __len__(self):
        """Return the number of triplets in the dataset."""
        return len(self.triples)

    def __getitem__(self, idx):
        """
        Load and return a triplet (hazy RGB, SAR, clear RGB) as tensors.
        Applies the provided transforms if available.
        """
        hp, sp, cp = self.triples[idx]
        hazy = Image.open(hp).convert("RGB")
        sar = Image.open(sp).convert("L")
        clear = Image.open(cp).convert("RGB")
        if self.transform_rgb:
            hazy = self.transform_rgb(hazy)
            clear = self.transform_rgb(clear)
        if self.transform_sar:
            sar = self.transform_sar(sar)
        return hazy, sar, clear