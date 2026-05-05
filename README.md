# Brain Tumor Classification

A deep learning project for multi-class brain tumor classification using brain MRI scans.

This project explores transfer learning, model evaluation, and interpretability techniques for medical image classification.


## Overview

The goal is to classify brain MRI images into four categories:

- Glioma
- Meningioma
- Pituitary
- No Tumor

The project focuses on:
- Transfer learning experiments
- Model comparison
- Interpretability analysis
- Failure-case analysis


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

- ResNet-18 (Frozen Backbone)
- ResNet-18 (Fine-tuned)
- ResNet-50 (Fine-tuned)

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

### ResNet-18 (Frozen)
- Accuracy: ~77.0%
- Macro F1 Score: ~76.2%
- Glioma Recall: ~70%

### ResNet-18 (Fine-tuned)
- Accuracy: ~95.3%
- Macro F1 Score: ~95.2%
- Glioma Recall: ~82%

### ResNet-50 (Fine-tuned)
- Accuracy: ~95.6%
- Macro F1 Score: ~95.5%
- Glioma Recall: ~84%

Key observations:
- Fine-tuning provides major performance gains over frozen transfer learning.
- ResNet-50 shows small but consistent improvements over ResNet-18.
- Performance improvements are most noticeable in harder classes like glioma.


## Transfer Learning Insights

Key findings:

- Frozen transfer learning is insufficient for this medical imaging task.
- Full fine-tuning significantly improves domain adaptation.
- Medical imaging benefits from feature adaptation beyond ImageNet pretraining.

Main insight:

MRI image representations differ substantially from natural image representations, making fine-tuning essential.


## Model Interpretability (Grad-CAM)

Grad-CAM is used to visualize model attention and understand prediction behavior.

Observations:
- ResNet-18 shows broader and more diffuse attention for glioma cases.
- ResNet-50 produces more localized activation patterns.
- Meningioma and pituitary consistently show focused discriminative regions.
- Failure-case analysis shows that false negatives often correspond to mislocalized attention.

Key insight:

Higher classification difficulty (especially glioma) correlates with less focused model attention, while deeper architectures improve localization quality.


## Failure Case Analysis

Failure-case analysis was performed on incorrectly classified samples.

Example:
- Glioma misclassified as No Tumor
- Grad-CAM showed mislocalized attention away from likely discriminative regions

Key insight:

Incorrect predictions often correspond to weak or misaligned visual attention.

## Running the Project

Install dependencies:

```bash
pip install -r requirements.txt
```

Train ResNet-18 (Frozen):

```bash
python src/training/train_resnet18_frozen.py
```

Train ResNet-18 (Fine-tuned):

```bash
python src/training/train_baseline.py
```

Train ResNet-50 (Fine-tuned):

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

- Experiment with Vision Transformers (ViT)
- Improve augmentation strategy for harder classes
- Experiment with class-weighted loss for glioma sensitivity
- Analyze more failure cases across all classes
- Compare higher-resolution inputs
- Build an agentic medical report generation pipeline