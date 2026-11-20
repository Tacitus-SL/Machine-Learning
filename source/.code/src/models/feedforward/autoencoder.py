import torch

class Autoencoder(torch.nn.Module):
    def __init__(
            self,
            channels,
            activation=torch.nn.ReLU):
        ...
        super().__init__()
        ## YOUR CODE HERE
        # -- start --
        encoder_layers = []
        for index in range(len(channels) - 2):
            encoder_layers.append(torch.nn.Linear(channels[index], channels[index + 1]))
            encoder_layers.append(activation())
        encoder_layers.append(torch.nn.Linear(channels[-2], channels[-1]))

        self.encoder = torch.nn.Sequential(*encoder_layers)

        decoder_layers = []
        decoder_channels = channels[::-1]
        for index in range(len(decoder_channels) - 2):
            decoder_layers.append(torch.nn.Linear(decoder_channels[index], decoder_channels[index + 1]))
            decoder_layers.append(activation())
        decoder_layers.append(torch.nn.Linear(decoder_channels[-2], decoder_channels[-1]))

        self.decoder = torch.nn.Sequential(*decoder_layers)
        # -- end --
        if not hasattr(self, 'encoder'):
            self.encoder = torch.nn.Linear(channels[0], channels[-1])
        if not hasattr(self, 'decoder'):
            self.decoder = torch.nn.Linear(channels[-1], channels[0])

    def __forward_kernel(self, signal):
        input_shape = signal.shape
        res = signal
        ## YOUR CODE HERE
        # -- start --
        res = res.reshape([res.shape[0], -1])
        res = self.encoder(res)
        res = self.decoder(res)
        # -- end --
        res = res.reshape(input_shape)
        return res

    def forward(self, batch):
        ## YOUR CODE HERE
        # -- start --
        reconstruction = self.__forward_kernel(batch['data']['image'])
        batch['signals'] = {'reconstruction': reconstruction}
        # -- end --
        if 'signals' not in batch:
            batch['signals'] = {'reconstruction': batch['data']['image']}
        return batch
