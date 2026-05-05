import torch
from torchvision import datasets, transforms
from torchvision.models import resnet18
from torch.utils.data import DataLoader

DEVICE = (
    "cuda" if torch.cuda.is_available()
    else "mps" if torch.backends.mps.is_available()
    else "cpu"
)

TEST_DIR = "data/Testing"
BATCH_SIZE = 32

test_transforms = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225]
    )
])

test_dataset = datasets.ImageFolder(TEST_DIR, transform=test_transforms)
test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False)

NUM_CLASSES = len(test_dataset.classes)

model = resnet18()
model.fc = torch.nn.Linear(
    model.fc.in_features,
    NUM_CLASSES
)

model.load_state_dict(
    torch.load("best_model.pth", map_location=DEVICE)
)

model = model.to(DEVICE)
model.eval()

def find_wrong_prediction():
    with torch.no_grad():
        for batch_idx, (images, labels) in enumerate(test_loader):
            images = images.to(DEVICE)

            outputs = model(images)
            preds = torch.argmax(outputs, dim=1)

            for i in range(len(preds)):
                if preds[i].item() != labels[i].item():
                    dataset_index = batch_idx * BATCH_SIZE + i
                    image_path, _ = test_dataset.samples[dataset_index]
                    
                    print("Image path:", image_path)
                    print("Wrong prediction found")
                    print("Predicted:", test_dataset.classes[preds[i].item()])
                    print("Actual:", test_dataset.classes[labels[i].item()])
                    return

if __name__ == "__main__":
    find_wrong_prediction()