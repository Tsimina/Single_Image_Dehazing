import os
import torch
import torch.optim as optim
from torch.utils.data import DataLoader, random_split
from torchvision import transforms
from models_ia2.aod_net import AODNet
from dataset import HazyClearDataset
import matplotlib.pyplot as plt
from skimage.metrics import peak_signal_noise_ratio as psnr
from skimage.metrics import structural_similarity as ssim
from multiprocessing import freeze_support
import numpy as np 

print('all modules imported')

if __name__ == '__main__':
    freeze_support()
    # Config
    DATA_ROOT = "D:\Ari"
    BATCH_SIZE = 16
    EPOCHS = 20
    LR = 1e-4
    VAL_SPLIT = 0.1
    SAVE_DIR = "saved_models"
    os.makedirs(SAVE_DIR, exist_ok=True)

    # Transforms
    transform = transforms.Compose([
        transforms.Resize((256, 256)),
        transforms.RandomHorizontalFlip(),
        transforms.RandomRotation(10),
        transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2),
        transforms.ToTensor(),
    ])

    # Dataset
    full_dataset = HazyClearDataset(
        hazy_dir='D:\Ari\hazy',
        clear_dir='D:\Ari\clear',
        transform=transform
    )

    print(f"Total samples found: {len(full_dataset)}")
    print(f"Sample paths: {full_dataset.valid_pairs[:3]}")

    # Train/Val Split
    val_size = int(VAL_SPLIT * len(full_dataset))
    train_size = len(full_dataset) - val_size
    train_dataset, val_dataset = random_split(full_dataset, [train_size, val_size])

    # DataLoaders
    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, 
                            num_workers=2, pin_memory=torch.cuda.is_available())
    val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False,
                          pin_memory=torch.cuda.is_available())

    # Model
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print('using - ', device)
     
    model = AODNet().to(device)
    optimizer = optim.Adam(model.parameters(), lr=LR)
    criterion = torch.nn.L1Loss()
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min', patience=3)

    train_losses = []
    train_psnrs = []
    train_ssims = []
    val_losses = []
    val_psnrs = []
    val_ssims = []

    # PSNR and SSIM per batch
    def calculate_metrics(dehazed, clean):
        dehazed = dehazed.detach().clamp(0, 1).cpu().numpy()
        clean = clean.cpu().numpy()
        batch_psnr = 0
        batch_ssim = 0
        for d, c in zip(dehazed, clean):
            d = np.transpose(d, (1, 2, 0))
            c = np.transpose(c, (1, 2, 0))
            batch_psnr += psnr(c, d, data_range=1.0)
            batch_ssim += ssim(c, d, data_range=1.0, channel_axis=-1)
        return batch_psnr/len(dehazed), batch_ssim/len(dehazed)

    # Training Loop
    for epoch in range(1, EPOCHS+1):
        model.train()
        tl, tp, ts = 0.0, 0.0, 0.0
        for hazy, clear in train_loader:
            hazy, clear = hazy.to(device), clear.to(device)
            optimizer.zero_grad()
            dehazed, _, _ = model(hazy)
            loss = criterion(dehazed, clear)
            loss.backward()
            optimizer.step()
            tl += loss.item()
            bpsnr, bssim = calculate_metrics(dehazed, clear)
            tp += bpsnr; ts += bssim
        train_losses.append(tl/len(train_loader))
        train_psnrs.append(tp/len(train_loader))
        train_ssims.append(ts/len(train_loader))

        model.eval()
        vl, vp, vs = 0.0, 0.0, 0.0
        with torch.no_grad():
            for hazy, clear in val_loader:
                hazy, clear = hazy.to(device), clear.to(device)
                dehazed, _, _ = model(hazy)
                loss = criterion(dehazed, clear)
                vl += loss.item()
                bpsnr, bssim = calculate_metrics(dehazed, clear)
                vp += bpsnr; vs += bssim
        val_losses.append(vl/len(val_loader))
        val_psnrs.append(vp/len(val_loader))
        val_ssims.append(vs/len(val_loader))

        print(f"Epoch {epoch}/{EPOCHS} | "
              f"Train Loss: {train_losses[-1]:.4f} | Val Loss: {val_losses[-1]:.4f} | "
              f"Train PSNR: {train_psnrs[-1]:.2f} dB | Val PSNR: {val_psnrs[-1]:.2f} dB | "
              f"Train SSIM: {train_ssims[-1]:.4f} | Val SSIM: {val_ssims[-1]:.4f}")
        
        scheduler.step(val_losses[-1])

    # Plotting
    plt.figure(figsize=(12, 4))
    plt.subplot(1, 3, 1)
    plt.plot(train_losses, label='Train Loss')
    plt.plot(val_losses, label='Val Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    plt.title('Training/Validation Loss')

    plt.subplot(1, 3, 2)
    # plt.plot(train_psnrs, label='Train PSNR')
    plt.plot(val_psnrs,   label='Val PSNR')
    plt.xlabel('Epoch')
    plt.ylabel('dB')
    plt.title('Validation PSNR')

    plt.subplot(1, 3, 3)
    # plt.plot(train_ssims, label='Train SSIM')
    plt.plot(val_ssims,   label='Val SSIM')
    plt.xlabel('Epoch')
    plt.ylabel('Score')
    plt.title('Validation SSIM')

    plt.tight_layout()
    plt.savefig('training_metrics_aod.png')
    plt.show()

    torch.save(model.state_dict(), f"aod_net_{EPOCHS}_epochs.pth")