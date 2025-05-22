# Single_Image_Dehazing


## Description

This project proposes a prior-guided deep learning framework for single-image dehazing, in which the Dark Channel Prior (DCP)—a well-established handcrafted heuristic—is embedded directly into a U-Net architecture. By incorporating the DCP map as an additional input channel, the network is guided by both physically motivated priors and learned convolutional representations. This hybrid design leverages the complementary strengths of classical image priors and data-driven feature extraction to enhance dehazing performance, particularly in challenging visibility conditions.

We also explore a multimodal variant that incorporates a synthetic SAR-like reflectivity map alongside the RGB and DCP inputs. This structural channel provides depth-aware guidance and helps the network disambiguate haze in regions such as the sky or low-texture surfaces, where handcrafted priors alone are often unreliable.

## Features 
- Prior-based dehazing models built on a U-Net backbone
- Incorporation of atmospheric scattering priors (e.g., dark channel prior)
- Load and apply pre-trained model weights
- Process individual images or batches
- Example script for quick testing on a single image

## Structure of the repo

```
Single_Image_Dehazing/
├── dataset/    # Dataloader script and image resizing utilities
├── model_configurations/      # Model definition 
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

Our prior-based U-Net models were trained on the RESIDE Beta dataset, using a total of 18,200 images covering diverse haze conditions and paired clean references. You can download the dataset here: [RESIDE Beta Dataset](https://utexas.app.box.com/s/25idwrsn890w03grdr6pls28cy38r91i). We used the OTS (Outdoor Training Set)for our application. 

![reside_dataset](https://github.com/user-attachments/assets/8dde7f4b-95e6-4abe-a596-4425c0ab5067)

In addition to the RGB images, we generate:

- **Dark Channel Prior (DCP) maps**, computed directly from the hazy images using a local minimum filter.
- **Synthetic SAR reflectivity maps**, created from the hazy images to simulate radar-like structural guidance. These maps are designed to mimic key SAR properties such as edge enhancement and speckle noise, and are used only in the `DCP+SAR Guided U-Net` model.

  ![sar_vs_gen](https://github.com/user-attachments/assets/0b24011d-9d2f-4e0e-8c72-b55c2a63acca)

  > Figure: Left – Real SAR image from Sentinel-1. Right – Synthetic SAR-like map generated from a hazy RGB input using our edge-based simulation pipeline.

To streamline data loading and ensure consistent alignment between modalities, we implemented a custom PyTorch `dataloader` that automatically maps each clear ground-truth image to its corresponding hazy version. While matching is performed based on consistent filename stems, allowing the loader to construct structured input triplets hazy, sar and their corresponding target clear without manual intervention.

  ![mapping](https://github.com/user-attachments/assets/b1ae3caa-eb3f-4a80-b04f-ab07ddf52737)

  ![sar_dcp](https://github.com/user-attachments/assets/ee87f39e-05f4-4f43-bbf1-3657bbc67ffb)


This modular loading scheme enables flexible switching between models: the DCP-only variant loads only hazy RGB and dark-channel maps, while the DCP+SAR configuration additionally includes the SAR reflectivity channel. The `dataloader` handles all input formatting and tensor concatenation required for multimodal training and evaluation.


## Model Details

This repository contains two main dehazing architectures:

- **U-Net DCP (vanilla):** A baseline U-Net model that uses the RGB input concatenated with a Dark Channel Prior (DCP) map to guide transmission estimation.


![dcp_vanilla](https://github.com/user-attachments/assets/54a213c4-d5d9-4d52-b706-df3aa08a414a)


- **U-Net DCP+SAR:** An extended multimodal variant that integrates a synthetic SAR reflectivity map alongside the RGB and DCP inputs, allowing the network to leverage both appearance-based and structure-aware priors for more robust dehazing, especially in challenging regions like sky or low-texture areas.

  
![dcp_sar_sch](https://github.com/user-attachments/assets/2e03974c-a54b-4e86-8e83-66e66113ded8)


Our dehazing network uses a U-Net architecture to fuse multi-scale feature representations with haze-specific priors:

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
python -m dataset.resize  --input dataset/ --output dataset/resized --width 256 --height 256
```
> [!NOTE] 
> The size argumet was set to 256 by default.

**SAR Image Generation**

Generate SAR maps for SAR-based models, using the hazy dataset:

```
python dataset/sar_img_gen.py <input_path> <output_path> --noise 25
```

> [!NOTE] 
> SAR maps are used alongside hazy/clean pairs during SAR-enhanced training.

# Training

```
cd Single_Image_Dehazing
DCP+SAR model
python -m src.train_dcp_sar

# U-Net with priors
python -m src.train_dcp_unet
```
> [!NOTE] 
> The training scripts use the dataloader.py and dataloader_sar.py  to map the hazy ground-thruth images correspodingly to their clear counterparts.

# Inference

```
#U-Net 
python src/inference.py --model test_application/saved_models/<model_path> --input_dir src/haze --output_dir src/results --model_type unet 

# DCP-SAR Guided
python src/inference.py --model test_application/saved_models/<model_path> --input_dir src/haze --output_dir src/results --model_type sar_unet --sar_dir src/sar
```
> [!NOTE] 
> For the inference sripts (specifically the SAR model) the image names should have the next structure <img_name>_sar. You cand also specify the extension of the image with the argument `--sar_ext` (the default being set to .png).


## Testing (Single Image)

**Standard**
For DCP U-Net vanilla architecture

```
cd test_application
python single_image_test.py --image <haze_img> --model saved_models/<model_name> --device cuda
```

**SAR-Enhanced**
For SAR enhanced DCP architecture

```
cd test_application
python single_image_test.py --image <haze_img> --model saved_models/<model_name> --use_sar  --sar_path <path_to_reflective_sar_map>
```
## Perfromance

For our experiments, we observed that the training curves tend to plateau around epoch 20, indicating convergence of both reconstruction loss and perceptual metrics such as PSNR and SSIM. All models were trained using an NVIDIA GTX 1660 Ti GPU, with a typical training duration of approximately 4, up to 6 hours when following our proposed methodology and input configuration. 

| Model                 | PSNR val      | SSIM val (%) | Inference Time (s) |
|:---------------------:|:-------------:|:------------:|:------------------:|
| DCP U-Net vanilla     |    31.1815    |    93,98     |       0.2165       |
| DCP-SAR Enhanced U-Net|     31.61     |    94.36     |       0.2599       |

Compared to earlier methods such as AOD-Net (29.07 / 86.41%) or Deep DCP (24.32  / 93.42%), our approach benefits from the explicit use of a handcrafted prior while still leveraging the representational power of convolutional networks. Although transformer-based models like DehazeFormer report higher PSNR scores (e.g., 37.54 ), they typically require substantially more computational resources.


## Examples 
An example result of our Single Image Dehazing pipeline. The left image is the original hazy photograph, and the right image demonstrates the dehazing result produced by the U-Net based prior model.
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





