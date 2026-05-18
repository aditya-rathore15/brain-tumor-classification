import torch
import torch.nn as nn
from torchvision import transforms, models
from PIL import Image

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


def predict_image(image_path, model):

    image = Image.open(image_path).convert("RGB")

    input_tensor = transform(image).unsqueeze(0).to(DEVICE)

    with torch.no_grad():
        outputs = model(input_tensor)

        probabilities = torch.softmax(outputs, dim=1)[0]

        predicted_index = torch.argmax(probabilities).item()

    prediction = CLASS_NAMES[predicted_index]

    confidence = probabilities[predicted_index].item()

    probability_dict = {
        CLASS_NAMES[i]: round(probabilities[i].item(), 4)
        for i in range(len(CLASS_NAMES))
    }

    return {
        "prediction": prediction,
        "confidence": round(confidence, 4),
        "probabilities": probability_dict
    }


if __name__ == "__main__":

    model = load_model()

    result = predict_image(
        "data/Testing/glioma/Te-gl_1.jpg",
        model
    )

    print(result)