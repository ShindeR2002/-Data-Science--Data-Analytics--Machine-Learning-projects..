# Neural network (NumPy implementation)

This folder contains a small From-Scratch neural-network implementation in NumPy and a short description.

Contents
- neural_network.py — a small educational neural network implementation (forward/backward, simple optimizers).
- README.md — this file (replaces `intro (1).md`)

Purpose
- Educational code to understand layer, activation, and optimizer implementations.
- Not optimized for production, but useful for learning and small experiments.

How to use
- Inspect `neural_network.py` to understand classes: Layer, FC_layer, AC_layer, Activation, Optimizer, Network.
- Example usage (pseudo):
  ```py
  from neural_network import Network, Activation
  net = Network()
  net.path([784, 128, 10])  # example: input 784 -> 128 -> 10
  net.fit(X_train, y_train, epoch=100, learning_rate=0.01, batch_size=64, optimizer=None, loss_type="Cross_entropy", regularization=None)
  preds = net.predict(X_test)
  ```

Notes & improvements
- The implementation uses simple NumPy operations and custom optimizers.
- Consider adding unit tests, clearer docstrings, and a small example script that runs a full train/eval on a toy dataset (I can add this if you want).
