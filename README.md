# Single_Image_Dehazing


## Description
This project implements a prior-based deep learning method for removing haze from a single input image. We leverage traditional image dehazing priors integrated into a UNet architecture, combining the strengths of classical priors with modern convolutional networks. The goal is to restore clarity, contrast, and color fidelity in photographs affected by natural or artificial fog.

## Features 
- Prior-based dehazing models built on a UNet backbone
- Incorporation of atmospheric scattering priors (e.g., dark channel prior)
- Load and apply pre-trained model weights
- Process individual images or batches
- Example script for quick testing on a single image

## Model Details

Our dehazing network uses a UNet architecture to fuse multi-scale feature representations with haze-specific priors:

- **Encoder:** Downsampling path extracts hierarchical features  
- **Decoder:** Upsampling path restores spatial resolution and refines the dehazed output  
- **Skip Connections:** Preserve fine details by connecting encoder and decoder layers  
- **Prior Integration:** Dedicated modules inject haze priors (e.g., transmission map estimates) into intermediate feature maps  


## Dataset 

Our prior-based UNet models were trained on the RESIDE Beta dataset, using a total of 18,200 images covering diverse haze conditions and paired clean references. You can download the dataset here: [RESIDE Beta Dataset](https://utexas.app.box.com/s/25idwrsn890w03grdr6pls28cy38r91i). We used the OTS (Outdoor Training Set)for our application. 

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

## Limitations

Due to limited hardware resources (e.g., GPU memory and compute), model training and inference were constrained, which may affect performance and image quality.





