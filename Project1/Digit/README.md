# Digit classification (MNIST)

Goal
Train a simple neural network (or use a small model) to classify MNIST digits and report accuracy.

Contents
- mnist_digit.ipynb — main notebook (renamed from `mnist_digit (1).ipynb`)
- README.md — this file

How to run
1. From Project1 folder create and activate a venv and install deps:
   python3 -m venv venv
   source venv/bin/activate
   pip install -r ../requirements.txt

2. Open the notebook:
   jupyter notebook mnist_digit.ipynb

Notes
- The notebook downloads MNIST automatically using the standard dataset utilities (e.g., torchvision or tensorflow datasets).
- Random seed used: 42 (documented in the notebook)
- Expected accuracy with the baseline small model: ~97–99% (depends on model and training)
