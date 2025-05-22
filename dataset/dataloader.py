# dataloader.py
import os
import glob
from PIL import Image
import torch
from torch.utils.data import Dataset, DataLoader, random_split
from torchvision import transforms

class MultiHazyDataset(Dataset):
    """
    root_dir/
        clear/ — ground truth (e.g. 3068.jpg)
        hazy/  — hazy (e.g. 3068_0.8_0.1.jpg)
    """
    def __init__(self, root_dir, transform=None, clear_ext="jpg", haze_exts=("jpg", "png")):
        clear_dir = os.path.join(root_dir, "clear_rsz")
        haze_dir  = os.path.join(root_dir, "hazy_rsz")
        assert os.path.isdir(clear_dir) and os.path.isdir(haze_dir), "Structura folderelor invalidă"
        self.transform = transform
        self.pairs = []
        # map clear stems to paths
        clear_map = {}
        for cp in glob.glob(os.path.join(clear_dir, f"*.{clear_ext}")):
            stem = os.path.splitext(os.path.basename(cp))[0]
            clear_map[stem] = cp
        # flatten hazy images
        for hp in glob.glob(os.path.join(haze_dir, "*_*.*")):
            stem = os.path.basename(hp).split('_')[0]
            if stem in clear_map:
                self.pairs.append((hp, clear_map[stem]))
        assert self.pairs, "Nu există nicio pereche validă"

    def __len__(self):
        return len(self.pairs)

    def __getitem__(self, idx):
        hp, cp = self.pairs[idx]
        hazy = Image.open(hp).convert("RGB")
        clear = Image.open(cp).convert("RGB")
        if self.transform:
            hazy = self.transform(hazy)
            clear = self.transform(clear)
        return hazy, clear


def get_loaders(root_dir, batch_size=8, val_ratio=0.2, test_ratio=0.1, num_workers=2, transform=None):

    ds = MultiHazyDataset(root_dir, transform)
    n = len(ds)
    n_val = int(val_ratio * n)
    n_test = int(test_ratio * n)
    n_train = n - n_val - n_test
    if n_train < 1: raise ValueError("Not enough data for training")
    train_ds, val_ds, test_ds = random_split(ds, [n_train, n_val, n_test])
    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True, num_workers=num_workers, pin_memory=True)
    val_loader   = DataLoader(val_ds,   batch_size=batch_size, shuffle=False, num_workers=num_workers, pin_memory=True)
    test_loader  = DataLoader(test_ds,  batch_size=batch_size, shuffle=False, num_workers=num_workers, pin_memory=True)
    return train_loader, val_loader, test_loader