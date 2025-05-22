# dataloader.py
import os
from torch.utils.data import Dataset, DataLoader, random_split
from torchvision import transforms
from PIL import Image

"""
Custom dataset for loading paired hazy and clear images.
Assumes directory structure:
    root_dir/
        hazy/
        clear/
"""
class HazyClearDataset(Dataset):
    def __init__(self, root_dir, transform=None):
        """
        root_dir: Path to the dataset root directory.
        transform: Optional torchvision transforms to apply to images, if they are not resized to 256x256
        """
        self.root_dir = root_dir
        self.transform = transform
        self.hazy_dir = os.path.join(root_dir, "hazy")
        self.clear_dir = os.path.join(root_dir, "clear")
        # List all image files in hazy directory
        self.hazy_files = sorted([
            f for f in os.listdir(self.hazy_dir)
            if os.path.isfile(os.path.join(self.hazy_dir, f))
        ])

    def __len__(self):
        """Return the number of samples in the dataset."""
        return len(self.hazy_files)

    def __getitem__(self, idx):
        """
        Load and return a sample (hazy image, clear image) as tensors.
        """
        hazy_path = os.path.join(self.hazy_dir, self.hazy_files[idx])
        clear_path = os.path.join(self.clear_dir, self.hazy_files[idx])
        hazy_img = Image.open(hazy_path).convert("RGB")
        clear_img = Image.open(clear_path).convert("RGB")
        if self.transform:
            hazy_img = self.transform(hazy_img)
            clear_img = self.transform(clear_img)
        return hazy_img, clear_img

"""
Helper function to create train, validation, and test DataLoaders.
Splits the dataset according to the given ratios.
"""
def get_loaders(root_dir, batch_size=4, val_ratio=0.2, test_ratio=0.1, num_workers=0, transform=None):
    """
    root_dir: Path to dataset root.
    batch_size: Batch size for DataLoaders.
    val_ratio: Fraction of data for validation.
    test_ratio: Fraction of data for testing.
    num_workers: Number of worker processes.
    transform: torchvision transforms to apply.
    Returns: train_loader, val_loader, test_loader
    """
    dataset = HazyClearDataset(root_dir, transform=transform)
    n = len(dataset)
    n_val = int(val_ratio * n)
    n_test = int(test_ratio * n)
    n_train = n - n_val - n_test
    train_ds, val_ds, test_ds = random_split(dataset, [n_train, n_val, n_test])
    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True, num_workers=num_workers)
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False, num_workers=num_workers)
    test_loader = DataLoader(test_ds, batch_size=batch_size, shuffle=False, num_workers=num_workers)
    return train_loader, val_loader, test_loader