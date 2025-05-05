import os
import argparse
import numpy as np
from PIL import Image
import torch
import torch.nn as nn
import torchvision.models as models
from torch.utils.data import Dataset, DataLoader
from sklearn.metrics import accuracy_score, precision_score, recall_score, roc_auc_score
import torchvision.transforms as transforms

# -----------------------------
# Preprocessing and Label Mapping
# -----------------------------
# Transformation used both in training and testing
transform = transforms.Compose([
    transforms.Grayscale(num_output_channels=1),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.5], std=[0.5])
])

# Global label mapping used during training.
# For each dataset, the raw label (0, 1, …) is mapped using these offsets.
DATASETS_DETAILS = {
    "octmnist": [0, 1, 2, 3],
    "pneumoniamnist": [4, 5],
    "breastmnist": [6, 7],
    "retinamnist": [8, 9, 10, 11, 12]
}

def map_label(raw_label, dataset_name):
    """
    Map a raw label from the test data to the global label using the dataset's offset mapping.
    Assumes raw_label is either a single number or an array-like with one element.
    """
    offsets = DATASETS_DETAILS[dataset_name]
    if isinstance(raw_label, (list, np.ndarray)):
        raw_val = int(raw_label[0])
    else:
        raw_val = int(raw_label)
    return offsets[raw_val]

# -----------------------------
# Data Preparation: Combine Test Data
# -----------------------------
def load_and_combine_test_data(dataset_folder):
    """
    The dataset folder should contain the following .npz files:
      - octmnist.npz
      - pneumoniamnist.npz
      - breastmnist.npz
      - retinamnist.npz
    Each file is expected to contain arrays named "test_images" and "test_labels".
    """
    combined_images = []
    combined_labels = []
    
    for dataset_name, offsets in DATASETS_DETAILS.items():
        npz_path = os.path.join(dataset_folder, f"{dataset_name}.npz")
        if not os.path.exists(npz_path):
            print(f"Warning: {npz_path} not found. Skipping {dataset_name}.")
            continue
        data = np.load(npz_path)
        test_images = data["test_images"]
        test_labels = data["test_labels"]
        
        # Process each sample: apply transform and map label
        for img, label in zip(test_images, test_labels):
            # Convert the raw image to a PIL image.
            # Assume the image is grayscale (2D) or RGB (3D). If RGB, it will be converted to grayscale.
            if img.ndim == 3 and img.shape[-1] == 3:
                img_pil = Image.fromarray(img.astype(np.uint8))
            elif img.ndim == 2:
                img_pil = Image.fromarray(img.astype(np.uint8))
            else:
                # Fallback in case the image format is unexpected.
                img_pil = Image.fromarray(img.astype(np.uint8))
            
            img_tensor = transform(img_pil)  # Convert to tensor and normalize.
            # Remove the extra channel dimension (to be added later in dataset class) if needed.
            combined_images.append(img_tensor.squeeze(0).numpy())
            combined_labels.append(map_label(label, dataset_name))
        
        print(f"Processed {dataset_name}: {len(test_images)} samples.")
    
    combined_images = np.array(combined_images)
    combined_labels = np.array(combined_labels, dtype=np.int64)
    print(f"\nCombined Test Dataset: {len(combined_images)} samples in total.")
    return combined_images, combined_labels

# -----------------------------
# Custom Dataset for Combined Test Data
# -----------------------------
class CustomDataset(Dataset):
    def __init__(self, images, labels):
        self.images = images  # Expecting numpy array of shape [N, H, W]
        self.labels = labels

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        # Add channel dimension to the image (1, H, W)
        img = np.expand_dims(self.images[idx], axis=0).astype(np.float32)
        img = torch.from_numpy(img)
        label = int(self.labels[idx])
        return img, label

# -----------------------------
# Model Definition
# -----------------------------
class ModifiedResNet18(nn.Module):
    def __init__(self, num_classes=13):
        super(ModifiedResNet18, self).__init__()
        self.resnet = models.resnet18()
        # Modify the first conv layer to accept 1-channel (grayscale) images.
        self.resnet.conv1 = nn.Conv2d(1, 64, kernel_size=7, stride=2, padding=3, bias=False)
        self.resnet.fc = nn.Linear(self.resnet.fc.in_features, num_classes)

    def forward(self, x):
        return self.resnet(x)

# -----------------------------
# Evaluation Function
# -----------------------------
def evaluate_model(model_path, model_name, test_loader, num_classes=13):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = ModifiedResNet18(num_classes=num_classes).to(device)
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.eval()

    all_preds, all_labels, all_probs = [], [], []
    with torch.no_grad():
        for images, labels in test_loader:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            probs = torch.softmax(outputs, dim=1)
            _, preds = torch.max(outputs, 1)

            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
            all_probs.extend(probs.cpu().numpy())

    accuracy = accuracy_score(all_labels, all_preds)
    precision = precision_score(all_labels, all_preds, average="weighted")
    recall = recall_score(all_labels, all_preds, average="weighted")
    auc = roc_auc_score(all_labels, all_probs, multi_class="ovo")

    print(f"\nEvaluation Results: {model_name}")
    print(f"Accuracy : {accuracy:.4f}")
    print(f"Precision: {precision:.4f}")
    print(f"Recall   : {recall:.4f}")
    print(f"AUC      : {auc:.4f}")

# -----------------------------
# Main
# -----------------------------
if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Combine test data from .npz files and evaluate one or more models."
    )
    parser.add_argument('--dataset_folder', type=str, required=True,
                        help="Path to the dataset folder containing npz files: octmnist.npz, pneumoniamnist.npz, breastmnist.npz, retinamnist.npz")
    parser.add_argument('--model_paths', type=str, nargs='+', required=True,
                        help="Paths to the model weights file(s) to evaluate.")
    parser.add_argument('--model_names', type=str, nargs='+', default=None,
                        help="Optional: Names for the models (should match the number of model_paths).")
    parser.add_argument('--batch_size', type=int, default=64,
                        help="Batch size for evaluation (default: 64).")
    parser.add_argument('--num_classes', type=int, default=13,
                        help="Number of classes (default: 13).")
    
    args = parser.parse_args()

    # Load and combine test data
    print(f"Loading test data from folder: {args.dataset_folder}")
    test_images, test_labels = load_and_combine_test_data(args.dataset_folder)
    
    # Create DataLoader
    test_dataset = CustomDataset(test_images, test_labels)
    test_loader = DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False)

    # Set model names if not provided
    if args.model_names is None or len(args.model_names) != len(args.model_paths):
        # Default names: Model 1, Model 2, ...
        model_names = [f"Model {i+1}" for i in range(len(args.model_paths))]
    else:
        model_names = args.model_names

    # Evaluate each model
    for model_path, model_name in zip(args.model_paths, model_names):
        print(f"\nEvaluating model: {model_name} from {model_path}")
        evaluate_model(model_path, model_name, test_loader, num_classes=args.num_classes)
