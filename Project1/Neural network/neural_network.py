import numpy as np
import matplotlib.pyplot as plt



import struct

def load_mnist_images(filename):
    with open(filename, 'rb') as f:
        magic, num, rows, cols = struct.unpack(">IIII", f.read(16))
        images = np.fromfile(f, dtype=np.uint8).reshape(num, rows * cols)
    return images / 255.0  # Normalized for better training performance

def load_mnist_labels(filename):
    with open(filename, 'rb') as f:
        magic, num = struct.unpack(">II", f.read(8))
        labels = np.fromfile(f, dtype=np.uint8)
    return labels






class Layer():
    def __init__(self):
        self.input_ = None 
        self.output_ = None 
        
    def forward_prop(self, input_):
        raise NotImplementedError
    
    def backward_prop(self, error, learning_rate, optimizer, t, regu):
        raise NotImplementedError
        
class Optimizer:
    def NAG(self, weights, prev_weights, gamma):
        weight_look_ahead = weights - gamma * prev_weights
        return weight_look_ahead

    def ADAGRAD(self, dw, prev_weights):
        return prev_weights + dw**2

    def RMSprop(self, dw, prev_weights, beta):
        return beta * prev_weights + (1 - beta) * dw**2

    def ADAM(self, dw, m, v, beta1, beta2):
        m = beta1 * m + (1 - beta1) * dw
        v = beta2 * v + (1 - beta2) * dw**2
        return m, v

class FC_layer(Layer):
    def __init__(self, input_size, output_size):
        np.random.seed(101)
        self.weights_ = np.random.randn(output_size, input_size) * np.sqrt(2 / input_size)
        self.bias_ = np.zeros((1, output_size))
        self.prev_weights_ = np.zeros_like(self.weights_)
        self.prev_bias_ = self.bias_.copy()
        self.m_w = self.prev_weights_.copy()
        self.v_w = self.prev_weights_.copy()
        self.m_b = self.prev_bias_.copy()
        self.v_b = self.prev_bias_.copy()
        
    def forward_prop(self, input_):
        self.input_ = input_
        self.output_ = self.input_ @ self.weights_.T + self.bias_
        return self.output_
    
    def backward_prop(self, error, learning_rate, optimizer, t, regu): 
        lamda, b = regu
        dw = error.T @ self.input_ + 2 * lamda * self.weights_
        db = np.sum(error, axis=0, keepdims=True)
        
        if optimizer == "NAG":
            gamma = 0.9
            weight_look_ahead = Optimizer().NAG(self.weights_, self.prev_weights_, gamma)
            input_error = error @ weight_look_ahead
            weights_update = gamma * self.prev_weights_ + dw * learning_rate
            bias_update = gamma * self.prev_bias_ + db * learning_rate
            self.weights_ -= weights_update
            self.bias_ -= bias_update
            self.prev_weights_ = weights_update
            self.prev_bias_ = bias_update
            
        elif optimizer == "ADAGRAD":
            tol = 1e-06
            input_error = error @ self.weights_
            self.prev_weights_ = Optimizer().ADAGRAD(dw, self.prev_weights_)
            self.prev_bias_ = Optimizer().ADAGRAD(db, self.prev_bias_)
            self.weights_ -= (learning_rate * dw) / (np.sqrt(self.prev_weights_ + tol))
            self.bias_ -= (learning_rate * db) / (np.sqrt(self.prev_bias_ + tol))
            
        elif optimizer == "RMSprop":
            tol = 1e-06
            beta = 0.95
            input_error = error @ self.weights_
            self.prev_weights_ = Optimizer().RMSprop(dw, self.prev_weights_, beta)
            self.prev_bias_ = Optimizer().RMSprop(db, self.prev_bias_, beta)
            self.weights_ -= (learning_rate * dw) / np.sqrt(self.prev_weights_ + tol)
            self.bias_ -= (learning_rate * db) / np.sqrt(self.prev_bias_ + tol)
        
        elif optimizer == "ADAM":
            beta1, beta2, tol = 0.9, 0.999, 1e-06
            input_error = error @ self.weights_
            self.m_w, self.v_w = Optimizer().ADAM(dw, self.m_w, self.v_w, beta1, beta2)
            self.m_b, self.v_b = Optimizer().ADAM(db, self.m_b, self.v_b, beta1, beta2)
            m_hat_w = self.m_w / (1 - beta1**t)
            v_hat_w = self.v_w / (1 - beta2**t)
            m_hat_b = self.m_b / (1 - beta1**t)
            v_hat_b = self.v_b / (1 - beta2**t)
            self.weights_ -= (learning_rate * m_hat_w) / (np.sqrt(v_hat_w + tol))
            self.bias_ -= (learning_rate * m_hat_b) / (np.sqrt(v_hat_b + tol))
            
        else: # Default SGD
            input_error = error @ self.weights_ 
            self.weights_ -= learning_rate * dw
            self.bias_ -= learning_rate * db
        
        return input_error

class AC_layer(Layer):
    def __init__(self, activation, activation_dr):
        self.act = activation
        self.act_dr = activation_dr
        
    def forward_prop(self, input_):
        self.input_ = input_
        self.output_ = self.act(self.input_)
        return self.output_
    
    def backward_prop(self, error, learning_rate, optimizer, t, regu):
        lamda, b = regu
        if b:
            roh = 0.05
            roh_hat = np.mean(self.output_, axis=0)
            return error * self.act_dr(self.input_) + b * (-(roh / roh_hat) + ((1 - roh) / (1 - roh_hat)))
        return error * self.act_dr(self.input_)

class Activation:
    def sigmoid(self, z):
        return 1 / (1 + np.exp(-np.clip(z, -500, 500)))

    def sigmoid_dr(self, x):
        s = self.sigmoid(x)
        return s * (1 - s)

    def softmax(self, z):
        # Subtract max for numerical stability
        shift_z = z - np.max(z, axis=1, keepdims=True)
        exps = np.exp(shift_z)
        return exps / np.sum(exps, axis=1, keepdims=True)

    def softmax_dr(self, z):
        s = self.softmax(z)
        return s * (1 - s)

    def tanh(self, x):
        return np.tanh(x)

    def tanh_prime(self, x):
        return 1 - np.tanh(x)**2
    
    def relu(self, x):
        return np.maximum(0, x)

    def relu_prime(self, x):
        return (x > 0).astype(float)

class Network():
    def __init__(self):
        self.layers = []
        self.error_list = []
        
    def add(self, layer):
        self.layers.append(layer)
        
    def error(self, X, y, batch_size, loss_type):
        if loss_type == "MSE":
            return 2 * (X - y) / batch_size
        return (X - y) / batch_size
        
    def path(self, neuron_info):
        for i in range(1, len(neuron_info)):
            self.add(FC_layer(neuron_info[i-1], neuron_info[i]))
            if i == len(neuron_info) - 1:
                self.add(AC_layer(Activation().softmax, Activation().softmax_dr))
            else:
                # NEW: Using ReLU for hidden layers
                self.add(AC_layer(Activation().relu, Activation().relu_prime))
    
    def fit(self, X, y, epoch, learning_rate, batch_size, optimizer, loss_type, regularization, l2_lamda=0, sparse_b=0):
        regu = (l2_lamda, sparse_b)
        m = X.shape[0]
        t = 0
        batch_no = max(1, m // batch_size)
        
        for i in range(epoch):
            for batch in range(batch_no):
                t += 1 # Corrected: increments per batch for ADAM
                str_idx, stp_idx = batch * batch_size, (batch + 1) * batch_size
                output = X[str_idx:stp_idx]
                for layer in self.layers:
                    output = layer.forward_prop(output)
                
                err = self.error(output, y[str_idx:stp_idx], batch_size, loss_type)
                for layer in reversed(self.layers):
                    err = layer.backward_prop(err, learning_rate, optimizer, t, regu)
                
            current_err = np.linalg.norm(self.predict(X) - y)
            self.error_list.append(current_err)
            if i > 0 and i % 5 == 0:
                learning_rate *= 0.8
                print(f"Learning rate decayed to: {learning_rate:.6f}")

    def predict(self, X):
        output = X
        for layer in self.layers:
            output = layer.forward_prop(output)
        return output

# --- EXECUTION BLOCK ---
# --- EXECUTION BLOCK ---
# --- FINAL EXECUTION AND REPORTING ---
if __name__ == "__main__":
    import os
    print("Initializing MNIST Training...")
    
    # 1. Paths to Kaggle cache
    download_path = "/home/codespace/.cache/kagglehub/datasets/hojjatk/mnist-dataset/versions/1"
    train_img_path = os.path.join(download_path, "train-images-idx3-ubyte", "train-images-idx3-ubyte")
    train_lbl_path = os.path.join(download_path, "train-labels-idx1-ubyte", "train-labels-idx1-ubyte")
    # NEW: Test set paths
    test_img_path  = os.path.join(download_path, "t10k-images-idx3-ubyte", "t10k-images-idx3-ubyte")
    test_lbl_path  = os.path.join(download_path, "t10k-labels-idx1-ubyte", "t10k-labels-idx1-ubyte")
    
    # 2. Load all data
    X_train = load_mnist_images(train_img_path)
    y_train_raw = load_mnist_labels(train_lbl_path)
    X_test = load_mnist_images(test_img_path) # Now X_test is defined!
    y_test_raw = load_mnist_labels(test_lbl_path) # Now y_test_raw is defined!
    
    # ... rest of your one-hot encoding and model.fit code ...
    
    # One-hot encoding
    y_train = np.zeros((y_train_raw.size, 10))
    y_train[np.arange(y_train_raw.size), y_train_raw] = 1

    # 2. Define and Train Model
    model = Network() # This defines 'model'
    model.path([784, 128, 10]) 
    
    model.fit(
        X_train, y_train, 
        epoch=20, 
        learning_rate=0.001, 
        batch_size=64, 
        optimizer="ADAM", 
        loss_type="Cross_entropy", 
        regularization=(0, 0)
    )

    # 3. Visualization Section (Now 'model' is guaranteed to exist)
    plt.figure(figsize=(10, 5))
    plt.plot(model.error_list, color='blue', linewidth=2)
    plt.title('Learning Curve: Model Error Over Time', fontsize=14)
    plt.xlabel('Epochs', fontsize=12)
    plt.ylabel('Error', fontsize=12)
    plt.grid(True, linestyle='--', alpha=0.6)
    plt.savefig("loss_curve.png")
    print("✅ Loss curve saved as loss_curve.png")
    plt.show()
    
    



## 1. The t-SNE Visualization
from sklearn.manifold import TSNE
import seaborn as sns

# We use the test set to see how well the model generalizes
sample_size = 2000
X_sample = X_test[:sample_size]
y_sample = y_test_raw[:sample_size]

# Feature extraction: Stopping at the last hidden layer
def get_hidden_features(X):
    output = X
    for layer in model.layers[:-2]: # Excludes the final output layer and its activation
        output = layer.forward_prop(output)
    return output

hidden_features = get_hidden_features(X_sample)

# Apply t-SNE to reduce 256 hidden dimensions to 2D
tsne = TSNE(n_components=2, random_state=101, init='pca', learning_rate='auto')
tsne_results = tsne.fit_transform(hidden_features)

plt.figure(figsize=(12, 8))
sns.scatterplot(
    x=tsne_results[:,0], y=tsne_results[:,1],
    hue=y_sample, palette=sns.color_palette("hls", 10),
    legend='full', alpha=0.7
)
plt.title('t-SNE Visualization: Model\'s Internal Representation of Digits', fontsize=15)
plt.show()