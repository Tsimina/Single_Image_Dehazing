import torch
import torch.nn as nn
import torch.nn.functional as F

""" 
DCP helper function.
Computes the dark channel prior for an input image tensor.
img_tensor: (B, 3, H, W) tensor
Returns: (B, 1, H, W) tensor representing the dark channel
"""
def get_dark_channel(img_tensor, window_size=15):
    min_rgb, _ = torch.min(img_tensor, dim=1, keepdim=True)  # Minimum across RGB channels
    dark = -F.max_pool2d(-min_rgb, kernel_size=window_size, stride=1, padding=window_size // 2)  # Local min pooling
    return dark

""" 
Basic UNet block: two convolutional layers with ReLU activations.
Used for both encoder and decoder paths.
"""
class UNetBlock(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        return self.block(x)

"""
DehazingUNet model.
UNet architecture that takes a 3-channel RGB image, computes its dark channel prior,
concatenates it as a 4th channel, and processes it through the UNet.
"""
class DehazingUNet(nn.Module):
    def __init__(self):
        super().__init__()
        # Encoder path
        self.enc1 = UNetBlock(4, 64)  # Input: RGB + DCP (4 channels)
        self.pool1 = nn.MaxPool2d(2)
        self.enc2 = UNetBlock(64, 128)
        self.pool2 = nn.MaxPool2d(2)
        self.enc3 = UNetBlock(128, 256)
        self.pool3 = nn.MaxPool2d(2)

        # Bottleneck
        self.bottleneck = UNetBlock(256, 512)

        # Decoder path
        self.up3 = nn.ConvTranspose2d(512, 256, kernel_size=2, stride=2)
        self.dec3 = UNetBlock(512, 256)
        self.up2 = nn.ConvTranspose2d(256, 128, kernel_size=2, stride=2)
        self.dec2 = UNetBlock(256, 128)
        self.up1 = nn.ConvTranspose2d(128, 64, kernel_size=2, stride=2)
        self.dec1 = UNetBlock(128, 64)

        # Final output layer
        self.final = nn.Conv2d(64, 3, kernel_size=1)

    def forward(self, x_rgb):
        """
        Forward pass for the DehazingUNet.
        x_rgb: (B, 3, H, W) input image tensor
        Returns: (B, 3, H, W) dehazed image tensor (values in [0,1])
        """
        x_dcp = get_dark_channel(x_rgb)  # Compute dark channel prior
        x = torch.cat([x_rgb, x_dcp], dim=1)  # Concatenate as 4th channel

        # Encoder
        enc1 = self.enc1(x)
        enc2 = self.enc2(self.pool1(enc1))
        enc3 = self.enc3(self.pool2(enc2))

        # Bottleneck
        bottleneck = self.bottleneck(self.pool3(enc3))

        # Decoder with skip connections
        dec3 = self.dec3(torch.cat([self.up3(bottleneck), enc3], dim=1))
        dec2 = self.dec2(torch.cat([self.up2(dec3), enc2], dim=1))
        dec1 = self.dec1(torch.cat([self.up1(dec2), enc1], dim=1))

        # Output
        return torch.sigmoid(self.final(dec1))