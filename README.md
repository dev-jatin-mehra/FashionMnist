# 🧥 FashionMNIST Classification with PyTorch

## 📌 Project Overview
This project implements a **simple feed-forward neural network** using PyTorch to classify images from the **FashionMNIST** dataset. The dataset contains **28x28 grayscale images** belonging to 10 different classes, representing various clothing items.

## 📂 Dataset Information
The **FashionMNIST dataset** consists of:
- **60,000 training images**
- **10,000 test images**
- Each image belongs to one of **10 classes**, including:
  - T-shirt/top
  - Trouser
  - Pullover
  - Dress
  - Coat
  - Sandal
  - Shirt
  - Sneaker
  - Bag
  - Ankle boot

## 📂Dataset Link
- https://www.kaggle.com/datasets/zalando-research/fashionmnist

## 📜 Model Architecture
The implemented **SimpleNet** is a **fully connected feed-forward neural network** with the following layers:
- **Input layer:** 28 × 28 neurons (flattened image input)
- **Hidden layers:**
  - First hidden layer: 128 neurons with **ReLU activation**
  - Second hidden layer: 64 neurons with **ReLU activation**
- **Output layer:** 10 neurons (one for each class)

```python
import torch.nn as nn
import torch.optim as optim

class SimpleNet(nn.Module):
    def __init__(self):
        super(SimpleNet, self).__init__()
        self.fc1 = nn.Linear(28 * 28, 128)  # Input layer
        self.fc2 = nn.Linear(128, 64)       # Hidden layer
        self.fc3 = nn.Linear(64, 10)        # Output layer (10 classes for FashionMNIST)

    def forward(self, x):
        x = x.view(x.shape[0], -1)  # Flatten the image
        x = torch.relu(self.fc1(x))  # First layer with ReLU
        x = torch.relu(self.fc2(x))  # Second layer with ReLU
        x = self.fc3(x)              # Output layer
        return x
```

## 🎯 Loss Function & Optimizer
- **Loss Function:** `CrossEntropyLoss()` (since it's a multi-class classification problem)
- **Optimizer:** `Adam()` with a learning rate of `0.001`

```python
# Instantiate the model
model = SimpleNet()

# Loss function and optimizer
criterion = nn.CrossEntropyLoss()  # Classification loss
optimizer = optim.Adam(model.parameters(), lr=0.001)  # Adam optimizer
```

## 📊 Training Pipeline
1. **Load the FashionMNIST dataset** (training and test data)
2. **Normalize the dataset** for better model performance
3. **Train the model** using mini-batch gradient descent
4. **Evaluate the model** on the test set

## 🚀 How to Run the Project
### **1️⃣ Install Dependencies**
```bash
pip install torch torchvision matplotlib
```

### **2️⃣ Train the Model**
Run the Python script to train the model:
```bash
python train.py
```

### **3️⃣ Evaluate the Model**
Once training is complete, evaluate the model on the test set:
```bash
python evaluate.py
```

## 📈 Expected Results
After training, the model should achieve an **accuracy of ~85%** on the test set. Further improvements can be made using techniques such as **dropout, batch normalization, and deeper architectures**.

## 🔥 Future Improvements
- Experiment with **CNNs (Convolutional Neural Networks)** for improved accuracy
- Implement a **FLASK API** for web realted tasks
