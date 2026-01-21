# Handwritten Digit Recognition: A From-Scratch Neural Engine
**Author:** Rohit, M.Tech Aerospace Engineering (IIT Kanpur)  
**Technical Focus:** Vectorized Linear Algebra, ADAM Optimization, and Manifold Analysis

---

## Section 1: Project Overview & Executive Summary

### 1.1 Problem Statement
In the field of computer vision, handwritten digit recognition (MNIST) serves as the primary benchmark for architectural efficiency. The objective was to design and implement a **Deep Neural Network (DNN)** built entirely from the ground up using **NumPy**, deliberately bypassing high-level frameworks like TensorFlow or PyTorch. This approach demonstrates a deep mastery of backpropagation mechanics and numerical stability.

### 1.2 Model Architecture Visualized
The network is structured as a feed-forward engine with the following topology:


![alt text](architecture.PNG)



### 1.3 Key Performance Indicators (KPIs)
The model underwent rigorous training and validation to achieve industry-standard performance:

* **Final Test Accuracy:** **96.62%**
* **Verification Accuracy:** **96.59%** (Secondary test on 10,000 unseen images)
* **Architectural Depth:** 3-Layer Fully Connected Network with 512 Hidden Neurons
* **Training Stability:** Successfully reached convergence over 40 epochs without gradient explosion.

---

## 1.4 Technical Architecture & Mathematics

### I. Model Topology
The architecture is designed to map high-dimensional pixel data into a categorical probability space:

* **Input Layer ($X$):** 784 units (Flattened 28x28 grayscale images).
* **Hidden Layer ($H_1$):** 512 neurons with **He Initialization** ($W \sim \mathcal{N}(0, \sqrt{2/n_{in}})$) to prevent signal saturation in the ReLU activation.
* **Output Layer ($Y$):** 10 neurons with **Softmax** activation for probability distribution.

### II. Optimization Stack: The ADAM Algorithm
To achieve fast convergence, I implemented the **ADAM Optimizer**, which utilizes adaptive moment estimation:


![alt text](adam_optimizer-1.PNG)



* **Momentum:** Implemented first-order ($m$) and second-order ($v$) moment estimations. 
* **Dropout Regularization:** Integrated a **10% Dropout** rate to force the network to learn robust feature representations.
* **Learning Rate Decay:** An 80% decay schedule every 5 epochs ensures the optimizer settles smoothly into the global minimum.

### III. Vectorized Computation
The engine is strictly vectorized using **NumPy** for maximum computational efficiency:

* **Batch Size:** 256 samples per iteration.
* **Forward Prop:** $Z = X \cdot W^T + b$.
* **Backward Prop:** Multi-layer chain rule implementation with L2 weight penalty.

---



## Section 2: Training Dynamics & Visual Analytics

### 2.1 Convergence Analysis
The **Convergence Plot** below captures the optimization journey of the model. Over 40 epochs, the error successfully decayed from approximately **10.1 to 6.5**.

* **ADAM Stability:** The smooth, consistent descent in the curve validates the effectiveness of the Adaptive Moment Estimation logic.
* **Optimization Proof:** The lack of major divergence in the later stages confirms that the **Learning Rate Decay** (80% every 5 epochs) successfully stabilized the weights as the model approached the global minimum.

![alt text](<convergence plot.PNG>)
**

---

### 2.2 Manifold Learning: t-SNE & PCA Clustering
To audit how the 512-neuron hidden layer interprets raw pixel data, I applied dimensionality reduction to project the internal features into a 2D plane.

* **t-SNE Visualization:** The t-SNE plot reveals that the model has successfully learned to group similar digits into distinct, well-separated geometric clusters.
* **Geometric Separation:** Simple, unique structures like **'0'** and **'1'** form isolated islands, directly correlating to their near-perfect recall rates.
* **Structural Proximity:** The proximity of clusters for **'4'**, **'7'**, and **'9'** visually explains the "Hard-to-Classify" edge cases where handwritten strokes share similar geometric properties.

![alt text](image.png)
**

---

### 2.3 Principal Component Analysis (PCA)
While t-SNE focuses on local neighbors, the **PCA dashboard** provides a global view of the feature variance.

* **Linear Separability:** The PCA results confirm that the network has transformed the original pixel space into a feature space where digits are significantly more separable, proving the hidden layer's role as a powerful feature extractor.

![alt text](image-1.png)





## Section 3: Diagnostic Auditing & Failure Analysis

### 3.1 Confusion Matrix: Performance by Digit
To evaluate the model beyond global accuracy, I generated a **Confusion Matrix** to audit the precision and recall of each digit class.

* **High-Precision Classes:** The model exhibits near-perfect recall for **'1'** (1,121 correct) and **'0'** (966 correct), indicating that these digits have the most distinct feature signatures.
* **Systematic Confusion:** The highest degree of error occurs between **'4'** and **'9'** (17 instances) and **'7'** and **'2'** (20 instances). This is expected given the topological similarities in handwritten variants of these digits.

![Confusion Matrix](image_1f4208.png)

---

### 3.2 Error Deep-Dive: Hard-to-Classify Digits
True engineering rigor involves looking at the "Success and Failure" images side-by-side.

* **Visual Test Success:** Random visual tests (such as digit '7') confirm that the model's prediction aligns with human visual perception.
* **Detailed Failure Analysis:** I extracted the specific images where the model failed. Many of these "misclassifications" involve digits with extreme human writing noise, such as a '5' that structurally resembles a '6' or a '9' that looks like an '8'.

![Error Analysis](image_1f4223.png)

---

### 3.3 Inference on Unseen Data
To verify generalizability, I visualized a grid of predictions on a randomized subset of the test data.

* **Generalization Proof:** The model correctly identifies various handwriting styles (slanted, thick, and thin strokes), proving that the **Dropout Regularization** and **L2 Penalty** successfully prevented overfitting.

![Model Predictions](image_1f41ec.png)