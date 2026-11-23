import torch

class Autoencoder(torch.nn.Module):
    def __init__(
            self,
            channels,
            activation=torch.nn.ReLU):
        super().__init__()
        encoder_layers = []
        for i in range(len(channels) - 1):
            encoder_layers.append(torch.nn.Linear(channels[i], channels[i + 1]))
            if i < len(channels) - 2:
                encoder_layers.append(activation())
        self.encoder = torch.nn.Sequential(*encoder_layers)

        decoder_layers = []
        for i in range(len(channels) - 1, 0, -1):
            decoder_layers.append(torch.nn.Linear(channels[i], channels[i - 1]))
            if i > 1:
                decoder_layers.append(activation())
        self.decoder = torch.nn.Sequential(*decoder_layers)

    def __forward_kernel(self, signal):
        input_shape = signal.shape
        res = signal.reshape(signal.shape[0], -1)
        res = self.encoder(res)
        res = self.decoder(res)
        res = res.reshape(input_shape)
        return res

    def forward(self, batch):
        image = batch['data']['image']
        reconstruction = self.__forward_kernel(image)
        batch['signals'] = {'reconstruction': reconstruction}
        return batch
