import os
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torchvision.models as models
from torch.utils.data import Dataset, DataLoader, Subset
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import accuracy_score
import torchvision.transforms as transforms
from PIL import Image
import shutil
import medmnist
from medmnist import INFO

# Set project directory to current working directory
project_dir = os.getcwd()
dataset_path = os.path.join(project_dir, "dataset")

# Create dataset directory if it does not exist
os.makedirs(dataset_path, exist_ok=True)

# List of all 4 datasets to download
DATASETS_LIST = [
    "octmnist", "pneumoniamnist", "breastmnist", "retinamnist"
]

# Download and save all datasets directly to the dataset folder
for dataset_name in DATASETS_LIST:
    dataset_class = getattr(medmnist, INFO[dataset_name]["python_class"])
    
    # Download train and validation splits into dataset_path
    dataset_class(split="train", download=True, root=dataset_path)
    dataset_class(split="val", download=True, root=dataset_path)
    
    print(f"{dataset_name} downloaded in {dataset_path}")

# Define transformations for preprocessing (convert to grayscale, normalize)
transform = transforms.Compose([
    transforms.Grayscale(num_output_channels=1),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.5], std=[0.5])
])

# Global label mapping (for reference)
global_label_mapping = {
    0: "choroidal neovascularization",
    1: "diabetic macular edema",
    2: "drusen",
    3: "normal (OCT)",
    4: "normal (Pneumonia)",
    5: "pneumonia",
    6: "malignant",
    7: "normal, benign (Breast)",
    8: "Retina 0",
    9: "Retina 1",
    10: "Retina 2",
    11: "Retina 3",
    12: "Retina 4"
}

# Dataset details with label offsets: (number_of_classes, label_offsets)
DATASETS_DETAILS = {
    "octmnist": (4, [0, 1, 2, 3]),
    "pneumoniamnist": (2, [4, 5]),
    "breastmnist": (2, [6, 7]),
    "retinamnist": (5, [8, 9, 10, 11, 12])
}

# Storage for combined images and labels
train_images_list, train_labels_list = [], []
val_images_list, val_labels_list = [], []

# Process each dataset
for dataset_name, (num_classes, label_offsets) in DATASETS_DETAILS.items():
    dataset_file = os.path.join(dataset_path, f"{dataset_name}.npz")
    
    # Load .npz file
    with np.load(dataset_file, allow_pickle=True) as data:
        train_images = data["train_images"]
        train_labels = data["train_labels"]
        val_images = data["val_images"]
        val_labels = data["val_labels"]
    
    # If images are in RGB, convert to grayscale by averaging channels
    if train_images.ndim == 4 and train_images.shape[-1] == 3:
        train_images = np.mean(train_images, axis=-1)
        val_images = np.mean(val_images, axis=-1)
    
    # Preprocess and collect training images and labels
    for img, label in zip(train_images, train_labels):
        img_pil = Image.fromarray(img.astype(np.uint8))
        img_tensor = transform(img_pil)
        class_index = label_offsets[int(label[0])]
        train_images_list.append(img_tensor.squeeze(0).numpy())  # Remove extra channel dim
        train_labels_list.append(class_index)
    
    # Preprocess and collect validation images and labels
    for img, label in zip(val_images, val_labels):
        img_pil = Image.fromarray(img.astype(np.uint8))
        img_tensor = transform(img_pil)
        class_index = label_offsets[int(label[0])]
        val_images_list.append(img_tensor.squeeze(0).numpy())
        val_labels_list.append(class_index)

# Convert lists to numpy arrays
train_images_array = np.array(train_images_list)
train_labels_array = np.array(train_labels_list, dtype=np.int64)
val_images_array = np.array(val_images_list)
val_labels_array = np.array(val_labels_list, dtype=np.int64)

# Save the combined dataset as a compressed npz file
np.savez_compressed(os.path.join(dataset_path, "combined_dataset.npz"),
                    train_images=train_images_array, train_labels=train_labels_array,
                    val_images=val_images_array, val_labels=val_labels_array)

print(f"Combined dataset saved at: {os.path.join(dataset_path, 'combined_dataset.npz')}")
print(f"Train Samples: {len(train_images_array)}, Validation Samples: {len(val_images_array)}")

# Define a custom dataset class for training
class MedicalDataset(Dataset):
    def __init__(self, images, labels):
        self.images = torch.tensor(images, dtype=torch.float32).unsqueeze(1)  # add channel dimension
        self.labels = torch.tensor(labels, dtype=torch.float32)

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        return self.images[idx], self.labels[idx]

# Create dataset objects for training and validation
train_dataset = MedicalDataset(train_images_array, train_labels_array)
val_dataset = MedicalDataset(val_images_array, val_labels_array)

# Define DataLoaders
train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=32, shuffle=False)

# Modify ResNet18 to accept grayscale images and output required number of classes
class ModifiedResNet18(nn.Module):
    def __init__(self, num_classes=13):
        super(ModifiedResNet18, self).__init__()
        self.resnet = models.resnet18(pretrained=True)
        self.resnet.conv1 = nn.Conv2d(1, 64, kernel_size=7, stride=2, padding=3, bias=False)
        self.resnet.fc = nn.Linear(self.resnet.fc.in_features, num_classes)

    def forward(self, x):
        return self.resnet(x)

# Device configuration
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Cross-validation settings
k_folds = 5
num_epochs = 10
batch_size = 64
num_classes = 13

# Combine train and validation arrays for cross-validation
full_images = np.concatenate([train_images_array, val_images_array], axis=0)
full_labels = np.concatenate([train_labels_array, val_labels_array], axis=0)

# Custom Dataset for the combined data
class CustomDataset(Dataset):
    def __init__(self, images, labels):
        self.images = images
        self.labels = labels

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        img = self.images[idx]
        label = int(self.labels[idx])
        img = np.expand_dims(img, axis=0).astype(np.float32)
        img = torch.from_numpy(img)
        return img, label

# Create full dataset for cross-validation
full_dataset = CustomDataset(full_images, full_labels)

# Setup cross-validation
skf = StratifiedKFold(n_splits=k_folds, shuffle=True, random_state=42)
best_acc = 0.0
best_model_state = None

for fold, (train_idx, val_idx) in enumerate(skf.split(full_images, full_labels)):
    print(f"\nFold {fold+1}/{k_folds}")
    
    # Create subset datasets
    train_subset = Subset(full_dataset, train_idx)
    val_subset = Subset(full_dataset, val_idx)
    
    train_loader = DataLoader(train_subset, batch_size=batch_size, shuffle=True, num_workers=2)
    val_loader = DataLoader(val_subset, batch_size=batch_size, shuffle=False, num_workers=2)
    
    # Initialize model, loss, and optimizer
    model = ModifiedResNet18(num_classes=num_classes).to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    
    # Training loop
    for epoch in range(num_epochs):
        model.train()
        running_loss = 0.0
        correct, total = 0, 0
        
        for images, labels in train_loader:
            images, labels = images.to(device), labels.to(device)
            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels.long())
            loss.backward()
            optimizer.step()
            
            running_loss += loss.item()
            _, predicted = torch.max(outputs, 1)
            correct += (predicted == labels.long()).sum().item()
            total += labels.size(0)
        
        train_acc = correct / total
        print(f"Epoch [{epoch+1}/{num_epochs}] Loss: {running_loss/len(train_loader):.4f}, Train Accuracy: {train_acc:.4f}")
    
    # Validation loop
    model.eval()
    all_preds, all_labels = [], []
    with torch.no_grad():
        for images, labels in val_loader:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            _, predicted = torch.max(outputs, 1)
            all_preds.extend(predicted.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
    
    val_acc = accuracy_score(all_labels, all_preds)
    print(f"Fold {fold+1} Validation Accuracy: {val_acc:.4f}")
    
    if val_acc > best_acc:
        best_acc = val_acc
        best_model_state = model.state_dict()

# Save best model weights in the 'model' directory
model_dir = os.path.join(project_dir, "model")
os.makedirs(model_dir, exist_ok=True)
best_model_path = os.path.join(model_dir, "model_weights.pth")
torch.save(best_model_state, best_model_path)
print(f"\nBest model (acc = {best_acc:.4f}) saved at: {best_model_path}")
