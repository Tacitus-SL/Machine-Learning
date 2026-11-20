import torch

class Autoencoder(torch.nn.Module):
    def __init__(
            self,
            channels,
            activation=torch.nn.ReLU):
        ...
        super().__init__()
        ## YOUR CODE HERE
        if not hasattr(self, 'encoder'):
            self.encoder = torch.nn.Linear(channels[0], channels[-1])
        if not hasattr(self, 'decoder'):
            self.decoder = torch.nn.Linear(channels[-1], channels[0])

    def __forward_kernel(self, signal):
        input_shape = signal.shape
        res = signal
        ## YOUR CODE HERE
        res = res.reshape(input_shape)
        return res

    def forward(self, batch):
        ## YOUR CODE HERE
        if 'signals' not in batch:
            batch['signals'] = {'reconstruction': batch['data']['image']}
        return batch
