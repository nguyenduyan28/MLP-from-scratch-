# MLP from Scratch – FashionMNIST Classifier

This project implements a Multi-Layer Perceptron (MLP) from scratch using NumPy, designed to classify fashion images from the Fashion-MNIST dataset. It reproduces key components of a deep learning framework without using PyTorch or TensorFlow.

# 🔧 Features
	•	Fully connected layers, ReLU activations, and forward propagation
	•	Manual backpropagation and gradient descent via Adam optimizer
	•	Cross-entropy loss function
	•	Custom Module, Linear, ReLU, and training loop logic
	•	Data loading, batching, and train/val/test split
	•	FashionMNIST dataset (images of clothing items in 10 classes)

# 🧠 Model Architecture

Input (28×28) → Flatten
→ Linear(784 → 512) → ReLU
→ Linear(512 → 256) → ReLU
→ Linear(256 → 10) → Softmax (via CrossEntropyLoss)

# 🚀 How to Run

1. Install Requirements

pip install numpy tqdm

2. Run Training

python main.py

The training pipeline uses your custom-built MLP and includes logging via tqdm.

# 📁 Project Structure

.
├── main.py             # Entry point, training/validation loop
├── nn.py               # Neural network building blocks (Linear, ReLU, Module)
├── loss.py             # Cross-entropy loss implementation
├── data.py             # Dataset, DataLoader, random_split
├── FashionMNIST.py     # Dataset downloading and preprocessing
├── MLP_pytorch.py      # Baseline model using PyTorch (for comparison)

# ✅ Results
	•	Accuracy: ~85–87% on Fashion-MNIST test set after 20 epochs
	•	Training fully on CPU using mini-batch SGD with Adam optimizer

# 📚 Notes
	•	No autograd used — gradients are computed and applied manually
	•	A PyTorch version is included in MLP_pytorch.py for performance comparison
