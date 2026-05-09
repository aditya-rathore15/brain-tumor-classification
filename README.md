# Brain Tumor Classification

A deep learning project for multi-class brain tumor classification using brain MRI scans.

This project explores transfer learning, architecture comparison, optimization strategies, interpretability techniques, and failure-case analysis for medical image classification.


## Overview

The goal is to classify brain MRI images into four categories:

- Glioma
- Meningioma
- Pituitary
- No Tumor

The project focuses on:

- Transfer learning experiments
- Architecture comparison
- Optimization experiments
- Model interpretability
- Failure-case analysis


## Dataset

Brain Tumor MRI Dataset (Kaggle)

Classes:

- Glioma
- Meningioma
- Pituitary
- No Tumor

Dataset split:

- Training set (80% train / 20% validation)
- Testing set


## Experimental Protocol

To ensure fair model comparison:

- Training data is split into training and validation subsets.
- Validation performance is used for model checkpoint selection.
- The test set remains untouched until final evaluation.

This avoids test-set leakage and provides more reliable performance estimates.


## Models Explored

Current experiments:

- ResNet-18 (Frozen Backbone)
- ResNet-18 (Fine-tuned)
- ResNet-50 (Fine-tuned)
- Vision Transformer (ViT-B/16)

Optimization experiments:

- Class-weighted loss (ResNet-50)

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

These metrics provide class-wise understanding beyond overall accuracy.


## Current Results

### ResNet-18 (Frozen)

- Accuracy: ~75.0%
- Macro F1 Score: ~74.0%
- Glioma Recall: ~61%

### ResNet-18 (Fine-tuned)

- Accuracy: ~93.0%
- Macro F1 Score: ~93.0%
- Glioma Recall: ~79%

### ResNet-50 (Fine-tuned) — Best Baseline

- Accuracy: ~94.0%
- Macro F1 Score: ~94.0%
- Glioma Recall: ~81%

### Vision Transformer (ViT-B/16)

- Accuracy: ~93.0%
- Macro F1 Score: ~93.0%
- Glioma Recall: ~79%

### ResNet-50 (Class-weighted Loss)

- Accuracy: ~94.0%
- Macro F1 Score: ~94.0%
- Glioma Recall: ~82%

Key observations:

- Fine-tuning significantly outperforms frozen transfer learning.
- ResNet-50 achieves the strongest baseline performance.
- Vision Transformer performs competitively but does not outperform CNN-based models.
- Class-weighted loss provides only marginal improvement in glioma sensitivity.
- Glioma remains the most challenging class across all architectures.


## Transfer Learning Insights

Key findings:

- Frozen transfer learning is insufficient for this medical imaging task.
- Full fine-tuning significantly improves domain adaptation.
- Medical imaging benefits from feature adaptation beyond ImageNet pretraining.

Main insight:

MRI image representations differ substantially from natural image representations, making fine-tuning essential.


## Architecture Insights

Key findings:

- Fine-tuned CNNs outperform frozen transfer learning by a large margin.
- ResNet-50 provides the strongest overall baseline.
- Vision Transformers perform competitively but do not surpass deeper CNNs on this dataset.
- CNN inductive bias remains effective for medical imaging tasks with limited data.

Main insight:

For small-to-medium medical imaging datasets, transfer-learned CNNs remain highly competitive against transformer-based architectures.


## Optimization Insights

Key findings:

- Class-weighted loss slightly improves glioma recall.
- The improvement is marginal and does not significantly change overall performance.
- The main classification bottleneck remains glioma detection.

Main insight:

Loss reweighting alone is insufficient to fully address hard-class sensitivity.


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

Train ResNet-50 (Class-weighted):

```bash
python src/training/train_resnet50_weighted.py
```

Train Vision Transformer:

```bash
python src/training/train_vit.py
```

Run Grad-CAM analysis:

```bash
python src/analysis/gradcam_analysis.py
```

Run failure-case analysis:

```bash
python src/analysis/error_analysis.py
```