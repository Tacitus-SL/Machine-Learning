import torch


class ResidualBottleneck(torch.nn.Module):
    def __init__(
            self, 
            in_channels,
            out_channels,
            prenormalization=lambda n_channels: torch.nn.Identity(),
            postnormalization=lambda n_channels: torch.nn.Identity(),
            activation=torch.nn.ReLU,
            compression=1,
            residual=True):

        super().__init__()
        self.block = torch.nn.Identity()
        self.bypass = torch.nn.Identity()
        self.residual = residual

        ## YOUR CODE HERE
        # -- start --
        stride = 1
        if out_channels != in_channels:
            stride = 2
            self.bypass = torch.nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=stride, padding=1)
        
        self.block = torch.nn.Sequential(
            prenormalization(in_channels),
            torch.nn.Conv2d(in_channels, out_channels // compression, kernel_size=3, padding=1),
            postnormalization(out_channels // compression),
            activation(),
            prenormalization(out_channels // compression),
            torch.nn.Conv2d(out_channels // compression, out_channels, kernel_size=3, stride=stride, padding=1),
            postnormalization(out_channels),
        )
        # -- end --

    def forward(self, signal):
        ## YOUR CODE HERE

        # -- start --
        if self.residual:
            signal = self.block(signal) + self.bypass(signal)
        else:
            signal = self.block(signal)
        # -- end --
        return signal


class FullyConvolutionalNN(torch.nn.Module):
    def __init__(
            self,
            block=lambda in_channels, out_channels: torch.nn.Conv2d(in_channels, out_channels, (1, 1)),
            in_channels=1,
            mid_channels=[16, 32, 64, 128],
            out_channels=10,
            n_blocks=[1, 1, 1, 1]):
        ...
        ## YOUR CODE HERE
        # Define network modules in the constructor

        # -- start --
        super().__init__()

        self.encoder = torch.nn.Conv2d(in_channels, mid_channels[0], kernel_size=3)
        

        blocks = []
        for lvl_index in range(len(mid_channels)):
            n_channels = mid_channels[lvl_index]
            for _ in range(n_blocks[lvl_index]):
                blocks.append(block(n_channels, n_channels))

            if lvl_index + 1 < len(mid_channels):
                blocks.append(block(n_channels, mid_channels[lvl_index + 1]))

        self.blocks = torch.nn.Sequential(*blocks)

        self.decoder = torch.nn.Linear(mid_channels[-1], out_channels)
        # -- end --

    def __forward_kernel(self, signal):
        ## YOUR CODE HERE
        # Pass the signal through the modules in forward

        # -- start --
        signal = signal.unsqueeze(1)
        signal = self.encoder(signal)
        signal = self.blocks(signal)
        signal = signal.mean(dim=-1).mean(dim=-1)
        signal = self.decoder(signal)
        # -- end --

        return signal

    def forward(self, batch):
        signal = batch['data']['image']
        signal = self.__forward_kernel(signal)

        # Put the result into the batch
        batch['signals'] = {'output': signal}

        # Perform postprocessing after we get the output
        self.postprocessing(batch)

        return batch

    def postprocessing(self, batch):

        # Take network's output from the batch
        signal = batch['signals']['output']

        ## YOUR CODE HERE
        # -- start --
        signal = signal.argmax(dim=1)
        # -- end --

        # Put the processed result into the batch
        batch['postprocessed'] = {'class': signal}
