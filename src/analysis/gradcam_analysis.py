import os
import cv2
import torch
import numpy as np
from torchvision import transforms
from torchvision.models import resnet18
from pytorch_grad_cam import GradCAM
from pytorch_grad_cam.utils.image import show_cam_on_image
from pytorch_grad_cam.utils.model_targets import ClassifierOutputTarget

DEVICE = (
    "cuda" if torch.cuda.is_available()
    else "mps" if torch.backends.mps.is_available()
    else "cpu"
)

OUTPUT_DIR = "outputs"
os.makedirs(OUTPUT_DIR, exist_ok=True)

test_transforms = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225]
    )
])

NUM_CLASSES = 4

model = resnet18()
model.fc = torch.nn.Linear(model.fc.in_features, NUM_CLASSES)

model.load_state_dict(torch.load("best_model.pth", map_location=DEVICE))

model = model.to(DEVICE)
model.eval()

target_layer = model.layer4[-1]

cam = GradCAM(
    model=model,
    target_layers=[target_layer]
)

def visualize_gradcam(image_path):
    img = cv2.imread(image_path)

    if img is None:
        raise ValueError(f"Image not found: {image_path}")

    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img = cv2.resize(img, (224, 224))

    rgb_img = img.astype(np.float32) / 255.0

    input_tensor = test_transforms(
    transforms.ToPILImage()(img.astype(np.float32))
    ).unsqueeze(0).to(DEVICE)

    output = model(input_tensor)
    pred_class = torch.argmax(output, dim=1).item()

    targets = [ClassifierOutputTarget(pred_class)]

    grayscale_cam = cam(
        input_tensor=input_tensor,
        targets=targets
    )[0]

    visualization = show_cam_on_image(
        rgb_img,
        grayscale_cam,
        use_rgb=True
    )

    save_path = os.path.join(OUTPUT_DIR, f"gradcam_{os.path.basename(image_path)}")

    cv2.imwrite(save_path, cv2.cvtColor(visualization, cv2.COLOR_RGB2BGR))

    print("Saved:", save_path)
    print("Predicted class:", pred_class)

if __name__ == "__main__":
    visualize_gradcam("data/Testing/glioma/Te-gl_1.jpg")
    "data/Testing/glioma/Te-gl_100.jpg"