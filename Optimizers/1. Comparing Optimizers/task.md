# Comparing Optimizers

Train the same small network on MNIST with full-dataset and mini-batch gradient descent, momentum, AdaGrad, Adam, Lion, and Muon, then add Lookahead, an exponential moving average of the weights, and reducing the learning rate on a plateau. You write gradient descent by hand once; every other method is its update formula in `source/code/src/optimizers.py`. Grading uses a tiny generated dataset and does not download data.
