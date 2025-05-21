# Single_Image_Dehazing


## Description
This project implements a prior-guided deep learning approach for single-image dehazing. We integrate the Dark Channel Prior (DCP) — a widely used handcrafted heuristic — directly into a U-Net architecture by embedding it as an additional input channel. This design allows the model to benefit from both traditional dehazing cues and learned convolutional features.

The goal is to restore visual clarity, contrast, and color fidelity in hazy or fog-degraded images, using a lightweight and interpretable architecture. Our method bridges classical priors with modern deep learning, improving performance especially in challenging regions where appearance-based cues alone may fail.

We also explore a multimodal variant that incorporates a synthetic SAR-like reflectivity map alongside the RGB and DCP inputs. This structural channel provides depth-aware guidance and helps the network disambiguate haze in regions such as the sky or low-texture surfaces, where handcrafted priors alone are often unreliable.

## Features 
- Prior-based dehazing models built on a UNet backbone
- Incorporation of atmospheric scattering priors (e.g., dark channel prior)
- Load and apply pre-trained model weights
- Process individual images or batches
- Example script for quick testing on a single image

## Structure of the repo

```
Single_Image_Dehazing/
├── dataset/    # Dataloader script and image resizing utilities
├── model/      # Model definition 
├── papers/     # Scientific articles and reference materials
├── research/   # Experiments, analyses, and useful links
├── src/        # Source code for training and inference
├── test_application/  # Test images, evaluation scripts and Pre-Trained models
|   |
|   ├── saved_models/ # Pre-Trained models that can be used to test the dehazing capabilities on the test/own images 
|
└── results/    # Performance metrics in text files
```




## Dataset 

Our prior-based UNet models were trained on the RESIDE Beta dataset, using a total of 18,200 images covering diverse haze conditions and paired clean references. You can download the dataset here: [RESIDE Beta Dataset](https://utexas.app.box.com/s/25idwrsn890w03grdr6pls28cy38r91i). We used the OTS (Outdoor Training Set)for our application. 

![reside_dataset](https://github.com/user-attachments/assets/8dde7f4b-95e6-4abe-a596-4425c0ab5067)

In addition to the RGB images, we generate:

- **Dark Channel Prior (DCP) maps**, computed directly from the hazy images using a local minimum filter.
- **Synthetic SAR reflectivity maps**, created from the hazy images to simulate radar-like structural guidance. These maps are designed to mimic key SAR properties such as edge enhancement and speckle noise, and are used only in the `UNet-DCP+SAR` model.

  ![sar_vs_gen](https://github.com/user-attachments/assets/0b24011d-9d2f-4e0e-8c72-b55c2a63acca)

  > Figure: Left – Real SAR image from Sentinel-1. Right – Synthetic SAR-like map generated from a hazy RGB input using our edge-based simulation pipeline.

  To streamline data loading and ensure consistent alignment between modalities, we implemented a custom PyTorch `dataloader` that automatically maps each clear ground-truth image to its corresponding hazy version and associated prior maps (e.g., dark-channel and synthetic SAR). File matching is performed based on consistent filename stems, allowing the loader to construct structured input triplets hazy, sar and their corresponding target clear without manual intervention.

  ![mapping](https://github.com/user-attachments/assets/b1ae3caa-eb3f-4a80-b04f-ab07ddf52737)

  ![sar_dcp](https://github.com/user-attachments/assets/ee87f39e-05f4-4f43-bbf1-3657bbc67ffb)


This modular loading scheme enables flexible switching between models: the DCP-only variant loads only RGB and dark-channel maps, while the DCP+SAR configuration additionally includes the SAR reflectivity channel. The `dataloader` handles all input formatting and tensor concatenation required for multimodal training and evaluation.

  
> Figure: Left – Real SAR image from Sentinel-1. Right – Synthetic SAR-like map generated from a hazy RGB input using our edge-based simulation pipeline.


All generated priors are saved and aligned with the input images to form multimodal input triplets: `[RGB_hazy, DCP, SAR]` for the SAR variant, and `[RGB_hazy, DCP]` for the baseline.

## Model Details

This repository contains two main dehazing architectures:

- **UNet-DCP (vanilla):** A baseline U-Net model that uses the RGB input concatenated with a Dark Channel Prior (DCP) map to guide transmission estimation.


![dcp_vanilla](https://github.com/user-attachments/assets/47d38053-9990-4b3e-a594-4740b6efd758)

- **UNet-DCP+SAR:** An extended multimodal variant that integrates a synthetic SAR reflectivity map alongside the RGB and DCP inputs, allowing the network to leverage both appearance-based and structure-aware priors for more robust dehazing, especially in challenging regions like sky or low-texture areas.

  
![dcp_sar_sch](https://github.com/user-attachments/assets/0fe46bac-d20c-4fb0-adae-8e253801f003)


Our dehazing network uses a UNet architecture to fuse multi-scale feature representations with haze-specific priors:

- **Encoder:** Downsampling path extracts hierarchical features  
- **Decoder:** Upsampling path restores spatial resolution and refines the dehazed output  
- **Skip Connections:** Preserve fine details by connecting encoder and decoder layers  
- **Prior Integration:** Dedicated modules inject haze priors (e.g., transmission map estimates) into intermediate feature maps 

## Requirements
- Python 3.7 or later  
- PyTorch  
- NumPy, OpenCV, Pillow  
- (Optional) CUDA for GPU acceleration



## Installation

**Clone the repository:**

```
git clone https://github.com/Tsimina/Single_Image_Dehazing.git
cd Single_Image_Dehazing
```


**Navigate to the repository**
```
cd Single_Image_Dehazing
 ```

**Create and activate a virtual environment (optional)**

```
python -m venv venv
source venv/bin/activate    # Linux/macOS
venv\Scripts\activate.bat   # Windows
```

**Install dependencies**
```
pip install -r requirements.txt
```


## Usage

**Preprocessing**

Resize dataset images to 256×256 to ensure consistency:

```
python -m dataset.resize 
  --input dataset/ 
  --output dataset/resized 
  --width 256
  --height 256
```
> [!NOTE] 
> The size argumet was set to 256 by default.

**SAR Image Generation**

Generate SAR maps for SAR-based models, using the hazy dataset:

```
python dataset/sar_img_gen.py
--input_dir dataset/hazy
--output_dir dataset/hazy_sar
```

> [!NOTE] 
> SAR maps are used alongside hazy/clean pairs during SAR-enhanced training.

# Training

```
cd Single_Image_Dehazing
DCP+SAR model
python -m src.train_dcp_sar

# UNet with priors
python -m src.train
```

> [!NOTE] 
> The training scripts use the dataloader.py and dataloader_sar.py  to map the hazy groundthruth images correspodingly to their clear counterparts.

## Testing (Single Image)

**Standard**
For DCP+UNet vanilla architecture

```
python single_image_test.py 
  --input test/haze_img2.jpg 
  --output results/dehazed_img2.jpg 
  --model model/unet_dehaze_prior.pth
```

**SAR-Enhanced**
For SAR enhanced DCP architecture

```
python test_application/single_image_test.py 
  --image test_application/haze_img1.jpg 
  --model test_application/dehazing_unet_sar_30.pth 
  --use_sar 
  --sar_path test_application/haze_img1_sar.jpg
```
## Perfromance

For our experiments, we observed that the training curves tend to plateau around epoch 20, indicating convergence of both reconstruction loss and perceptual metrics such as PSNR and SSIM. All models were trained using an NVIDIA GTX 1660 Ti GPU, with a typical training duration of approximately 4, up to 6 hours when following our proposed methodology and input configuration. 

| Model                 | PSNR val (dB) | SSIM val (%) |
|:---------------------:|:-------------:|:------------:|
| DCP+UNet vanilla      |    31.1815    |    93,98     |
| DCP+SAR Enhanced UNet |     31.61     |    94.36     |


## Examples 
An example result of our Single Image Dehazing pipeline. The left image is the original hazy photograph, and the right image demonstrates the dehazing result produced by the UNet-based prior model.
![dcp_unet](https://github.com/user-attachments/assets/95c456a1-970c-4d14-b212-6400c5a04cb4)


For SAR-enhanced dehazing, the model leverages the additional SAR map (middle) to improve haze estimation and enhance detail recovery.

![dcp_unet](https://github.com/user-attachments/assets/e8852539-a54b-4209-9a4a-f293a96a5c60)

> [!IMPORTANT]  
> To evaluate the `DCP-SAR` model, you need not only the hazy RGB input image, but also its corresponding **synthetic SAR-like reflectivity map**. This additional channel provides structural guidance during inference and must be precomputed for each test image.


Due to the characteristics of the dataset, which contains relatively few examples with extremely dense haze, the model tends to generalize better to scenes with moderate atmospheric degradation. As a result, in challenging test cases, the dehazing is significantly more effective in the foreground—where contrast and structure are more pronounced—while distant background regions, such as the sky or horizon, remain partially veiled or exhibit residual artifacts. This behavior can be observed in the qualitative example, where the subject and immediate surroundings are clearly restored, while distant elements are only partially recovered.


In scenes where haze is distributed more uniformly and without extreme density gradients, the model performs noticeably better. The dehazed output shows improved overall clarity, with fewer visible artifacts and more natural color reconstruction. In such cases, transmission estimation is more stable across the image, leading to smoother results and better preservation of global structure. As illustrated in the example, both foreground and background regions are effectively restored, and block artifacts are significantly reduced.

![uniform haze](https://github.com/user-attachments/assets/87d6101a-b6ea-4c16-abcd-f9bc19116025)


## Limitations

Due to limited hardware resources (e.g., GPU memory and compute), model training and inference were constrained, which may affect performance and image quality.





