# Brain Tumor Classification

A deep learning project for multi-class brain tumor classification using brain MRI scans.

This project explores transfer learning, model evaluation, and interpretability techniques for medical image classification.


## Overview
The goal is to classify brain MRI images into four categories:
- Glioma
- Meningioma
- Pituitary
- No Tumor

The project focuses on building a strong baseline, analyzing model behavior, and improving interpretability.


## Dataset

Brain Tumor MRI Dataset (Kaggle)

Classes:
- Glioma
- Meningioma
- Pituitary
- No Tumor

Dataset split:
- Training set
- Testing set

## Baseline Model

Current baseline:
- Backbone: ResNet-18 (ImageNet pretrained)
- Transfer Learning: Full fine-tuning
- Framework: PyTorch

Training setup:
- Image size: 224×224
- Batch size: 32
- Optimizer: Adam
- Learning rate: 1e-4
- Loss: CrossEntropyLoss
- Epochs: 5


## Evaluation Metrics

The model is evaluated using:
- Accuracy
- Precision
- Recall
- F1 Score
- Confusion Matrix
- Classification Report

These metrics provide better class-wise understanding beyond overall accuracy.

## Current Results

Baseline (ResNet-18):
- Accuracy: ~95.3%
- Macro F1 Score: ~95.2%

Class-wise observations:
- Pituitary shows the strongest performance
- Meningioma shows strong performance with focused localization
- Glioma remains the most challenging class


## Model Interpretability (Grad-CAM)

Grad-CAM is used to visualize model attention and understand prediction behavior.

Observations:
- Meningioma shows strong localized activation
- Pituitary shows meaningful regional attention
- Glioma shows broader and more diffuse activation patterns

Key insight:

Although the model achieves high classification accuracy (~95%), Grad-CAM visualizations suggest that harder classes (especially glioma) correspond to less focused model attention.