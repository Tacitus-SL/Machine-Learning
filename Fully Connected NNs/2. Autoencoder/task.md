# Autoencoder on MNIST

In the previous chapter, we assembled the common training pieces for a classifier: dataset, model, loss dictionary, optimizer, scheduler, dataloaders, MLflow logger, and the shared training loop.

In this task, we will reuse that infrastructure for a different goal. Instead of predicting a digit label, an autoencoder receives an image, compresses it into an embedding, and reconstructs the same image from that embedding.
