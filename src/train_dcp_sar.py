import torch
import torch.nn as nn
from torch.utils.data import DataLoader, random_split
from torchvision import transforms
from model.dcp_sar_unet import DehazingUNet
from dataset.dataloader_sar import MultiHazySARDataset
from PIL import Image
import os
import numpy as np
from skimage.metrics import peak_signal_noise_ratio, structural_similarity
import matplotlib.pyplot as plt

def tensor_to_numpy(tensor):
    if tensor.dim() == 4:
        tensor = tensor[0]
    return tensor.permute(1,2,0).cpu().numpy()

# Transorm for RGB și SAR
transform_rgb = transforms.Compose([
    transforms.Resize((256, 256)),
    transforms.ToTensor(),
])
transform_sar = transforms.Compose([
    transforms.Resize((256, 256)),
    transforms.ToTensor(),
])

# SAR dataset
dataset = MultiHazySARDataset(
    root_dir="dataset/images",
    transform_rgb=transform_rgb,
    transform_sar=transform_sar
)

# Split train/val/test
n = len(dataset)
val_ratio = 0.2
test_ratio = 0.1
n_val = int(val_ratio * n)
n_test = int(test_ratio * n)
n_train = n - n_val - n_test
train_ds, val_ds, test_ds = random_split(dataset, [n_train, n_val, n_test])
print(f"Train: {len(train_ds)}, Val: {len(val_ds)}, Test: {len(test_ds)}")

train_loader = DataLoader(train_ds, batch_size=4, shuffle=True, num_workers=0)
val_loader = DataLoader(val_ds, batch_size=4, shuffle=False, num_workers=0)
test_loader = DataLoader(test_ds, batch_size=4, shuffle=False, num_workers=0)

# Display a sample triple (hazy, sar, clear)
sample_hazy, sample_sar, sample_clear = dataset[0]
print("Hazy shape:", sample_hazy.shape)
print("SAR shape:", sample_sar.shape)
print("Clear shape:", sample_clear.shape)

fig, axs = plt.subplots(1, 3, figsize=(12, 4))
axs[0].imshow(sample_hazy.permute(1, 2, 0))
axs[0].set_title("Hazy RGB")
axs[1].imshow(sample_sar.squeeze(0), cmap='gray')
axs[1].set_title("SAR (simulat)")
axs[2].imshow(sample_clear.permute(1, 2, 0))
axs[2].set_title("Clear RGB")
for ax in axs:
    ax.axis("off")
plt.tight_layout()
plt.show()

# Model
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = DehazingUNet(use_sar=True).to(device)
optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)
criterion = nn.L1Loss()

results_path = "training_results_sar.txt"
if not os.path.exists(results_path):
    with open(results_path, "w") as f:
        f.write("=== Training Metrics per Epoch ===\n")

def evaluate_model(loader):
    model.eval()
    total_loss = 0.0
    psnr_sum = 0.0
    ssim_sum = 0.0
    mse_sum = 0.0
    mae_sum = 0.0
    count = 0
    with torch.no_grad():
        for hazy, sar, clear in loader:
            hazy, sar, clear = hazy.to(device), sar.to(device), clear.to(device)
            output = model(hazy, sar)
            loss = criterion(output, clear)
            total_loss += loss.item()
            output = output.detach().cpu()
            clear = clear.cpu()
            batch_size = output.shape[0]
            for i in range(batch_size):
                out_np = tensor_to_numpy(output[i])
                gt_np = tensor_to_numpy(clear[i])
                psnr_sum += peak_signal_noise_ratio(gt_np, out_np, data_range=1.0)
                ssim_sum += structural_similarity(gt_np, out_np, channel_axis=2, data_range=1.0)
                mse_sum += np.mean((gt_np - out_np) ** 2)
                mae_sum += np.mean(np.abs(gt_np - out_np))
                count += 1
    avg_loss = total_loss / len(loader)
    return {
        'Loss': avg_loss,
        'PSNR': psnr_sum/count if count else 0,
        'SSIM': ssim_sum/count if count else 0,
        'MSE': mse_sum/count if count else 0,
        'MAE': mae_sum/count if count else 0
    }

# Saved model path contains the number of epochs as extension
epochs = 20
for epoch in range(epochs):
    model.train()
    total_loss = 0.0
    for hazy, sar, clear in train_loader:
        hazy, sar, clear = hazy.to(device), sar.to(device), clear.to(device)
        output = model(hazy, sar)
        loss = criterion(output, clear)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
    avg_loss = total_loss / len(train_loader)

    # Evaluare pe val și test
    val_metrics = evaluate_model(val_loader)
    test_metrics = evaluate_model(test_loader)

    print(f"Epoch {epoch+1} Train Loss: {avg_loss:.4f} ")

# Scrie doar rezultatele finale (ultima epocă)
with open(results_path, "a") as f:
    f.write(f"=== Final results ===\n")
    f.write(f"Train Loss: {avg_loss:.4f}\n")
    f.write(f"Val Loss: {val_metrics['Loss']:.4f}\n")
    f.write(f"Test Loss: {test_metrics['Loss']:.4f}\n")
    f.write(f"Val PSNR: {val_metrics['PSNR']:.2f} | Val SSIM: {val_metrics['SSIM']:.4f} | "
            f"Val MSE: {val_metrics['MSE']:.6f} | Val MAE: {val_metrics['MAE']:.6f}\n")
    f.write(f"Test PSNR: {test_metrics['PSNR']:.2f} | Test SSIM: {test_metrics['SSIM']:.4f} | "
            f"Test MSE: {test_metrics['MSE']:.6f} | Test MAE: {test_metrics['MAE']:.6f}\n\n")

torch.save(model.state_dict(), "dehazing_unet_sar_20.pth")
print("Model saved: dehazing_unet_sar_20.pth")