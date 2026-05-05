# Brain Tumor Classification

A deep learning project for multi-class brain tumor classification using brain MRI scans.

This project explores transfer learning, model evaluation, and interpretability techniques for medical image classification.

## Overview

The goal is to classify brain MRI images into four categories:

- Glioma
- Meningioma
- Pituitary
- No Tumor

The project focuses on building strong baselines, comparing model architectures, and analyzing model behavior through interpretability techniques.

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


## Models Explored

Current experiments:

- ResNet-18 (ImageNet pretrained, full fine-tuning)
- ResNet-50 (ImageNet pretrained, full fine-tuning)

Framework:
- PyTorch

Training setup:
- Image size: 224×224
- Batch size: 32
- Optimizer: Adam
- Learning rate: 1e-4
- Loss: CrossEntropyLoss
- Epochs: 5


## Evaluation Metrics

The models are evaluated using:

- Accuracy
- Precision
- Recall
- F1 Score
- Confusion Matrix
- Classification Report

These metrics provide better class-wise understanding beyond overall accuracy.

## Current Results

### ResNet-18
- Accuracy: ~95.3%
- Macro F1 Score: ~95.2%
- Glioma Recall: ~82%

### ResNet-50
- Accuracy: ~95.6%
- Macro F1 Score: ~95.5%
- Glioma Recall: ~84%

Key observations:
- ResNet-50 shows a small but consistent improvement over ResNet-18.
- Performance gains are most noticeable on glioma cases.
- Pituitary remains the strongest-performing class across both models.


## Model Interpretability (Grad-CAM)

Grad-CAM is used to visualize model attention and understand prediction behavior.

Observations:
- ResNet-18 shows broader and more diffuse attention for glioma cases.
- ResNet-50 produces more localized activation patterns.
- Meningioma and pituitary consistently show focused tumor-relevant regions.
- Failure-case analysis shows that false negatives often correspond to mislocalized attention.

Key insight:

Higher classification difficulty (especially glioma) correlates with less focused model attention, while deeper architectures improve localization quality.


## Failure Case Analysis

Failure-case analysis was performed on incorrectly classified samples.

Example observation:
- A glioma sample was misclassified as no-tumor.
- Grad-CAM visualization showed mislocalized attention away from the tumor region.

This suggests that incorrect predictions often arise when the model focuses on non-discriminative regions.


## Running the Project

Install dependencies:

```bash
pip install -r requirements.txt
```

Train ResNet-18:

```bash
python src/training/train_baseline.py
```

Train ResNet-50:

```bash
python src/training/train_resnet50.py
```

Run Grad-CAM analysis:

```bash
python src/analysis/gradcam_analysis.py
```

Run failure-case analysis:

```bash
python src/analysis/error_analysis.py
```

---

## Future Work

- Compare frozen vs fine-tuned ResNet-18
- Experiment with Vision Transformers (ViT)
- Improve augmentation strategy for harder classes
- Experiment with class-weighted loss for glioma sensitivity
- Analyze more failure cases across all classes
- Build an agentic medical report generation pipeline
