# Multi-Dataset Medical Image Classification using Modified ResNet18

This project builds a unified image classification model on multiple medical datasets using PyTorch and the MedMNIST collection. It trains a modified ResNet18 model using 5-fold stratified cross-validation.

## 🛠️ Setup Instructions

### ✅ Prerequisites

Install the required Python packages:

```bash
pip install torch torchvision numpy scikit-learn pillow medmnist
```
--- 

## 🧠 Class Labels

The model outputs 13 classes, each mapped from the respective dataset:

| Class Index | Label Description                         |
|-------------|--------------------------------------------|
| 0           | Choroidal Neovascularization (OCT)         |
| 1           | Diabetic Macular Edema (OCT)               |
| 2           | Drusen (OCT)                               |
| 3           | Normal (OCT)                               |
| 4           | Normal (Pneumonia)                         |
| 5           | Pneumonia                                  |
| 6           | Malignant (Breast)                         |
| 7           | Benign/Normal (Breast)                     |
| 8-12        | Retina Classes 0-4                         |

---

## 📁 File: `main_model.py`

### 🚀 What This Script Does

1. **Downloads 4 MedMNIST datasets**:
   - OCTMNIST
   - PneumoniaMNIST
   - BreastMNIST
   - RetinaMNIST

2. **Processes and merges** all datasets into a combined training and validation dataset with relabeled classes.

3. **Defines a custom PyTorch Dataset** and applies image preprocessing (grayscale, normalization).

4. **Modifies ResNet18** to support single-channel grayscale images and outputs 13 unique medical condition classes.

5. **Trains the model using 5-fold cross-validation**, logging performance metrics per fold.

6. **Saves the best-performing model** (based on validation accuracy) to the `model/model_weights.pth` path.

## ▶️ How to Run

Run the script directly:

```bash
python main_model.py
```
---

## 📁 File: `test_model.py`

### 🔍 What This Script Does

This script loads the saved model and evaluate it on a **combined test set** derived from the same 4 datasets:

1. **Loads and combines test data** from the `.npz` files of:
   - OCTMNIST  
   - PneumoniaMNIST  
   - BreastMNIST  
   - RetinaMNIST  

2. **Maps labels globally** using a consistent index across all datasets (same as in training).

3. **Defines a custom PyTorch Dataset** for test data.

4. **Loads model weights**, modifies ResNet18 for grayscale input, and runs predictions.

5. **Calculates evaluation metrics**:
   - Accuracy  
   - Precision  
   - Recall  
   - ROC AUC Score (multi-class)

### 🧪 How to Run

```bash
python test_model.py \
    --dataset_folder path/to/dataset/folder \
    --model_paths model/model_weights.pth \
    --model_names "ResNet18 Combined" \
    --batch_size 64 \
    --num_classes 13
```

---

## 📦 Model Weights (5-Fold Cross-Validation)

The file **model_weights.pth** contains **ResNet18 model weights** obtained using **5-fold cross-validation**.

This model was trained on the combined datasets and validated using 5-fold cross-validation, with the **best-performing weights saved** in this file.

### ✅ How to Use the Model for Evaluation

To evaluate the model on your test data:

```bash
python test_model.py \
    --dataset_folder path/to/dataset/folder \
    --model_path model_weights.pth \
    --model_name "ResNet18 (5-Fold CV)" \
    --batch_size 64 \
    --num_classes 13
```

---

## 📓 notebook.ipynb

The file `notebook.ipynb` contains the **complete training pipeline for 6 ResNet18 models**, categorized as:

- **With Gaussian Blur Preprocessing** (3 models)
- **Without Preprocessing (Raw Data)** (3 models)

Each category includes models trained using:

1. **Train-only data**
2. **Train + fine-tune on validation data**
3. **5-Fold Cross-Validation** using both train and validation splits

This notebook handles **data loading, model training, evaluation**, and **comparison of all six models**. It provides a comprehensive view of model performance under different strategies and preprocessing techniques.

### 🛠️ How to Use

Run **notebook.ipynb** in a Jupyter environment (or Colab with necessary setup) to:

- Train all models from scratch
- Analyze evaluation metrics
- Compare preprocessing impacts and validation strategies

---

## 💻 ResNet18.ipynb

The file **ResNet18.ipynb** is designed for **training a single ResNet18 model from scratch** using:

- **Raw MedMNIST dataset (no preprocessing)**
- **5-Fold Cross-Validation** on the combined training and validation data

This notebook is optimized for **Google Colab** and focuses solely on the third model (no-blur, 5-fold CV), including:

- Data loading and transformation
- Training and validation loop with k-fold cross-validation
- Final test evaluation using MedMNIST datasets.

### 🚀 How to Run on Google Colab

1. Upload **ResNet18.ipynb** to [Google Colab](https://colab.research.google.com)
2. Run all cells to:
   - Train the ResNet18 model from scratch with 5-fold CV
   - Evaluate on the MedMNIST test data
   - View performance metrics

> This notebook does **not use pre-trained weights**. It builds the model from scratch for 5-fold evaluation.
