import torch
import numpy as np
import matplotlib.pyplot as plt
from torch.optim import Adam
from torch.nn import L1Loss
from skimage.metrics import peak_signal_noise_ratio, structural_similarity
from torchvision import transforms
from dataset.dataloader import get_loaders
from model_configurations.dcp_unet import DehazingUNet
from skimage.metrics import mean_squared_error

""" 
Define the image transformation pipeline for RGB images.
Converts images to PyTorch tensors.
"""
transform = transforms.Compose([
    transforms.ToTensor(),
])

"""
Load the training, validation, and test data loaders.
Splits the dataset according to the specified ratios.
"""
train_loader, val_loader, test_loader = get_loaders(
    root_dir="dataset/images", batch_size=4, val_ratio=0.2, test_ratio=0.1,
    num_workers=0, transform=transform
)

"""
Print the number of samples in each split for verification.
"""
print(f" train: {len(train_loader.dataset)}")
print(f" val:   {len(val_loader.dataset)}")
print(f" test:  {len(test_loader.dataset)}")

"""
Visualize a sample hazy and clear image from the training set.
"""
hazy_batch, clear_batch = next(iter(train_loader))
hazy_img = hazy_batch[0].permute(1,2,0).cpu().numpy()
clear_img = clear_batch[0].permute(1,2,0).cpu().numpy()
plt.figure(figsize=(8,4))
plt.subplot(1,2,1)
plt.title('Hazy')
plt.axis('off')
plt.imshow(hazy_img)
plt.subplot(1,2,2)
plt.title('Clear')
plt.axis('off')
plt.imshow(clear_img)
plt.show()

"""
Initialize the model, optimizer, and loss function.
Move the model to GPU if available.
"""
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = DehazingUNet().to(device)
optimizer = Adam(model.parameters(), lr=1e-4)
criterion = L1Loss()

"""
Helper function to convert a tensor to a NumPy array for visualization or metric computation.
Handles both batch and single image tensors.
"""
def tensor_to_numpy(tensor):
    # Batch
    if tensor.dim() == 4:
        tensor = tensor[0]  
    return tensor.permute(1,2,0).cpu().numpy()

"""
Evaluate the model on a given data loader.
Computes PSNR, SSIM, MSE, and MAE metrics for the dataset.
"""
def evaluate_model(loader):
    model.eval()
    psnr_sum = 0.0
    ssim_sum = 0.0
    mse_sum = 0.0
    mae_sum = 0.0
    count = 0
    with torch.no_grad():
        for hazy, clear in loader:
            hazy, clear = hazy.to(device), clear.to(device)
            out = model(hazy)
            # Suportă batch-uri > 1
            batch_size = hazy.shape[0]
            for i in range(batch_size):
                out_np = tensor_to_numpy(out[i])
                clear_np = tensor_to_numpy(clear[i])
                psnr_sum += peak_signal_noise_ratio(clear_np, out_np, data_range=1.0)
                ssim_sum += structural_similarity(clear_np, out_np, channel_axis=2, data_range=1.0)
                mse_sum += np.mean((clear_np - out_np) ** 2)
                mae_sum += np.mean(np.abs(clear_np - out_np))
                count += 1
    return {
        'PSNR': psnr_sum/count if count else 0,
        'SSIM': ssim_sum/count if count else 0,
        'MSE': mse_sum/count if count else 0,
        'MAE': mae_sum/count if count else 0
    }

"""
Main training loop.
Trains the model for a specified number of epochs and prints the average loss per epoch.
"""
epochs = 20
for epoch in range(1, epochs+1):
    model.train()
    total_loss = 0.0
    for hazy, clear in train_loader:
        hazy, clear = hazy.to(device), clear.to(device)
        out = model(hazy)
        loss = criterion(out, clear)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
    avg_loss = total_loss/len(train_loader)
    print(f"Epoch {epoch}/{epochs} – Loss: {avg_loss:.4f}")

"""
Evaluate the model on the training and validation sets after training.
Save the metrics to a text file.
"""
train_metrics = evaluate_model(train_loader)
val_metrics = evaluate_model(val_loader)

with open('training_results_dcp_sar_unet.txt', 'w') as f:
    f.write('=== Training and Validation Metrics ===\n')
    f.write(f"Train - Loss: {avg_loss:.4f}\n")
    for name, value in train_metrics.items():
        f.write(f"Train {name}: {value:.4f}\n")
    for name, value in val_metrics.items():
        f.write(f"Val {name}: {value:.4f}\n")
print("Metrics saved to training_results.txt")

"""
Save the trained model weights to a file.
"""
torch.save(model.state_dict(), "model_dcp_unet.pth")