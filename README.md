# Pneumonia Detection from Chest X-Ray Images

## Overview

This project implements a **Deep Learning model for detecting Pneumonia from chest X-ray images** using **PyTorch**.
The model learns to classify medical images into two categories:

* **Normal**
* **Pneumonia**

The notebook demonstrates the full deep learning pipeline including **data preprocessing, model training, validation, and evaluation**.

This project is useful for demonstrating how **computer vision and deep learning can assist in medical image analysis**.

---

## Dataset

The project uses the **Chest X-Ray Pneumonia dataset**.

Dataset structure:

```
chest_xray/
│
├── train/
│   ├── NORMAL/
│   └── PNEUMONIA/
│
├── val/
│   ├── NORMAL/
│   └── PNEUMONIA/
│
└── test/
    ├── NORMAL/
    └── PNEUMONIA/
```

Each image is a labeled chest X-ray belonging to one of the two classes.

---

## Technologies Used

* Python
* PyTorch
* Torchvision
* NumPy
* OpenCV
* Matplotlib
* PIL (Python Imaging Library)

---

## Project Workflow

### 1. Data Loading

The dataset is loaded using **PyTorch's `ImageFolder`**, which automatically assigns labels based on folder names.

---

### 2. Image Preprocessing

Images are transformed before training using the following steps:

* Resize images to **224 × 224**
* Convert images to tensors
* Normalize pixel values using **ImageNet statistics**

Example transformation pipeline:

```python
from torchvision import transforms

transform = transforms.Compose([
    transforms.Resize((224,224)),
    transforms.ToTensor(),
    transforms.Normalize(
        mean=[0.485,0.456,0.406],
        std=[0.229,0.224,0.225]
    )
])
```

---

### 3. DataLoader Creation

DataLoaders are used to efficiently load images in batches during training.

```python
from torch.utils.data import DataLoader

train_loader = DataLoader(train_dataset, batch_size=16, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=16)
test_loader = DataLoader(test_dataset, batch_size=16)
```

---

### 4. Model Architecture

A **Convolutional Neural Network (CNN)** is used to extract spatial features from the X-ray images.

Typical components include:

* Convolution layers
* ReLU activation
* MaxPooling layers
* Fully connected layers
* Softmax or LogSoftmax output

The network learns visual patterns that distinguish **healthy lungs from pneumonia infections**.

---

### 5. Training

The model is trained using:

* **Loss Function:** CrossEntropyLoss
* **Optimizer:** Stochastic Gradient Descent (SGD) or Adam
* **Batch Training**

Example training components:

```python
import torch
import torch.nn as nn
import torch.optim as optim

criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)
```

Training consists of:

1. Forward pass
2. Loss calculation
3. Backpropagation
4. Parameter update

---

### 6. Evaluation

The model is evaluated on the **validation and test datasets**.

Metrics include:

* Accuracy
* Loss
* Model predictions

---

### 7. Visualization

Training results are visualized using **Matplotlib**.

Examples include:

* Training vs Validation loss
* Training vs Validation accuracy

These graphs help monitor learning progress and detect **overfitting or underfitting**.

---

## How to Run the Project

### 1. Install Dependencies

```bash
pip install torch torchvision numpy matplotlib opencv-python pillow
```

---

### 2. Download the Dataset

Example using KaggleHub:

```python
import kagglehub

path = kagglehub.dataset_download("paultimothymooney/chest-xray-pneumonia")
print("Dataset path:", path)
```

---

### 3. Run the Notebook

Open the notebook:

```
pneumonia_detection.ipynb
```

Run all cells sequentially to train and evaluate the model.

---

## Example Output

The trained model predicts whether a chest X-ray corresponds to:

* **Normal lung condition**
* **Pneumonia infection**

Prediction results can be used to assist in **automated medical screening systems**.

---

## Possible Improvements

Future improvements could include:

* Using **Transfer Learning (ResNet, EfficientNet, DenseNet)**
* Adding **data augmentation**
* Performing **hyperparameter tuning**
* Implementing **model deployment using a web interface**
* Evaluating additional metrics such as **Precision, Recall, and F1-score**

---

## Author

Computer Engineering Student
Interested in **Artificial Intelligence, Machine Learning, and Computer Vision**
