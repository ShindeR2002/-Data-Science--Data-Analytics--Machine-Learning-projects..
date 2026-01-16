

Digit recognition and neural network experiments.

This folder contains two subprojects:
- Project1/Digit — MNIST digit classification experiments and notebooks.
- Project1/Neural network — a small educational neural network implementation (pure NumPy).

Overview
- Goal: Provide runnable notebooks and supporting scripts to reproduce the experiments.
- Structure is intentionally simple to make it easy for others to run locally.

Quickstart
1. Change to the project folder:
   cd Project1
2. Create a virtual environment and install dependencies:
   python3 -m venv venv
   source venv/bin/activate
   pip install -r requirements.txt
3. Open the digit notebook:
   jupyter notebook Digit/mnist_digit.ipynb
4. Or run the training placeholder:
   python src/train.py --help

Structure
- Digit/ — notebooks and docs for MNIST experiments
- Neural network/ — NumPy neural network implementation and docs
- src/ — small helper scripts (preprocessing, training)
- requirements.txt — Python dependencies for the project
- .gitignore — ignores for Python/Jupyter and data/models

Data
- MNIST is not included in the repo. The notebook uses the standard dataset loaders to fetch it automatically.

License & contribution
- No license file added per your instruction. If you want an MIT license, tell me and I will add it.
