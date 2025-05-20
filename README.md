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

The dataloader.py script creates a mapping between the clear images and the different 

## Structure of the repo

dataset: Dataloader script that matches ground-truth images with their corresponding hazy images, plus image resizing scripts.

model: Model definition scripts and trained model weights, usable by single_image_test.py.

papers: Collection of scientific articles and reference materials.

research: Experiments, analyses, and useful links.

src: Source code for training and inference implementations.

test: Test images and the test harness script for evaluating models.

results: Text files (.txt) containing model performance metrics.

## Requirements
- Python 3.7 or later  
- PyTorch  
- NumPy, OpenCV, Pillow  
- (Optional) CUDA for GPU acceleration






## Usage

## Examples 
![dcp_unet](https://github.com/user-attachments/assets/95c456a1-970c-4d14-b212-6400c5a04cb4)




