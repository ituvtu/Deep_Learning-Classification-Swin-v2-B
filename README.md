# README: Swin-V2-Base for AquaMonitor JYU Dataset

## Overview

This project focuses on the automatic classification of aquatic macroinvertebrates using deep learning to enhance environmental biomonitoring. Two neural network architectures were used: **Swin-V2-Base** and **ResNet18**. **Swin-V2-Base** achieved a **77% weighted F1-score**. This is an academic project aimed at evaluating the effectiveness of modern models in macroinvertebrate classification.

## Project Goal

Accurate monitoring of aquatic macroinvertebrates is crucial for assessing water quality and biodiversity. Manual identification requires significant time and specialized knowledge, making it difficult to scale. **Deep learning** can automate this process, improving both speed and accuracy.

## Importance and Relevance

- **Ecological Context**: Insect biomass has declined by **75%** over the past 30 years, making monitoring more crucial than ever.
- **Challenges of Manual Identification**: High costs, a limited number of experts, and scalability issues.
- **Deep Learning Advancements**: Modern models achieve expert-level accuracy under laboratory conditions.

## Dataset Description

**AquaMonitor JYU** is a subset of the large AquaMonitor dataset, which contains images of aquatic macroinvertebrates.

- **Number of Classes**: 31
- **Training Set**: 40,880 images (from 1,049 individuals)
- **Validation Set**: 6,394 images (from 157 individuals)
- **Test Set**: Hidden
- **Image Format**: 256x256

### Class Examples
To better understand the task's complexity, below are sample images from all 31 classes:

![Class examples](image.png)

### Class Imbalance
The dataset exhibits **significant class imbalance**, with the number of images per class ranging from **400 to 3,500**. This poses challenges during model training, as rare classes may lack sufficient representation for effective generalization.

Below is a histogram showing the distribution of classes in the training set:

![The distribution of classes](image-1.png)

## Swin-V2-Base Architecture

Swin-V2-Base is a transformer-based architecture that utilizes **hierarchical representation** and **local windows**, allowing it to efficiently process high-resolution images. However, the model showed **signs of overfitting**, emphasizing the importance of **pretraining** and regularization techniques.

### Model Configuration
- **Image Size**: 256x256
- **Number of Classes**: 31
- **Optimizer**: AdamW
- **Loss Function**: CrossEntropyLoss with label smoothing = 0.1
- **Epochs**: 6
- **Batch Size**: 64
- **Max Learning Rate**: 5e-5
- **Regularization**: Drop Path Rate = 0.2

### Weight Freezing
To improve model training:
- **Patch embedding layer weights were frozen**
- **Parameters of the first two layers were frozen**

### Training Setup
- **OneCycleLR** was used for adaptive learning rate scheduling.
- Model was trained on **A100 GPU in Google Colab** with **FP16 mixed precision**.
- Gradient scaling was performed using **torch.amp.GradScaler**.

### Model download
The Swin-V2-Base model can be downloaded at the following link:
[Download model.pt](https://www.dropbox.com/scl/fi/imlg8647aogsg0qzvwsv4/model.pt?rlkey=6t8y91cs6727ec4zsb935kit9&st=u89yv675&dl=0)


### Data Augmentation
Two augmentation strategies were applied:
1. **Standard augmentation for all images**:
   - Random rotation (10°)
   - Color jitter (brightness, contrast, saturation, hue)
   - Random resized cropping
   - Gaussian blur
   - Random affine transformations
2. **Stronger augmentation for rare classes**:
   - Random horizontal flip
   - Random perspective distortion
   - Stronger brightness and contrast adjustments

### Model Files
The repository contains:
- **model.pt** – The trained model checkpoint
- **model.py** – The model class for loading the trained model

## Results

| Architecture  | Weighted F1-score |
| ------------ | ----------------- |
| Swin-V2-Base | **77%**           |
| ResNet18     | 74%               |

The **Swin-V2-Base** model exhibited **overfitting tendencies**, whereas ResNet18 had lower performance overall.


## Conclusions

- **Swin-V2-Base achieved a 77% weighted F1-score**, but overfitting was observed.
- **Transfer learning is crucial**, as using pretrained models significantly improves results.
- **Possible improvements**: Increasing generalization by **data augmentation**, **regularization**, and alternative training strategies.

## License

This project is an academic study and is intended for research purposes only.

# Deep_Learning-Classification-Swin-v2-B
