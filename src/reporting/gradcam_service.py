import os

import cv2
import numpy as np
import torch
import torch.nn as nn

from PIL import Image

from torchvision import transforms, models

from pytorch_grad_cam import GradCAM
from pytorch_grad_cam.utils.image import show_cam_on_image


DEVICE = "mps" if torch.backends.mps.is_available() else "cpu"

CLASS_NAMES = [
    "glioma",
    "meningioma",
    "notumor",
    "pituitary"
]

transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
])


def load_model(model_path="best_resnet50_model.pth"):

    model = models.resnet50(weights=None)

    num_features = model.fc.in_features

    model.fc = nn.Linear(num_features, 4)

    model.load_state_dict(
        torch.load(model_path, map_location=DEVICE)
    )

    model.to(DEVICE)

    model.eval()

    return model


def determine_attention_quality(confidence):

    if confidence >= 0.90:
        return "localized"

    elif confidence >= 0.70:
        return "moderate"

    return "diffuse"


def generate_gradcam(
    image_path,
    model,
    confidence,
    output_dir="outputs/reports"
):

    os.makedirs(output_dir, exist_ok=True)

    image = Image.open(image_path).convert("RGB")

    rgb_image = np.array(image.resize((224, 224))) / 255.0

    input_tensor = transform(image).unsqueeze(0).to(DEVICE)

    target_layers = [model.layer4[-1]]

    cam = GradCAM(
        model=model,
        target_layers=target_layers
    )

    grayscale_cam = cam(
        input_tensor=input_tensor
    )[0]

    visualization = show_cam_on_image(
        rgb_image,
        grayscale_cam,
        use_rgb=True
    )

    filename = os.path.basename(image_path)

    output_path = os.path.join(
        output_dir,
        f"gradcam_{filename}"
    )

    cv2.imwrite(
        output_path,
        cv2.cvtColor(visualization, cv2.COLOR_RGB2BGR)
    )

    attention_quality = determine_attention_quality(
        confidence
    )

    return {
        "gradcam_path": output_path,
        "attention_quality": attention_quality
    }


if __name__ == "__main__":

    model = load_model()

    result = generate_gradcam(
        image_path="data/Testing/notumor/Te-no_10.jpg",
        model=model,
        confidence=0.8426
    )

    print(result)