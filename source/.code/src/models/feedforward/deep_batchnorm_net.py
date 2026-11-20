import torch

class BatchNorm(torch.nn.Module):
    """Normalize every channel to mean 0 and standard deviation 1.

    In training, the mean and the (biased) variance of the batch normalize
    the signal, and ``running_mean`` and ``running_var`` keep their
    exponential moving averages: ``running <- beta * running + (1 - beta) *
    batch``. In evaluation, the running averages are used instead:
    ``(signal - running_mean) / (sqrt(running_var) + eps)``.
    """

    def __init__(self, beta=0.90, eps=1.0e-4):
        super().__init__()
        self.beta = beta
        self.eps = eps
        self.register_buffer('running_mean', None)
        self.register_buffer('running_var', None)

    def _init_stats(self, signal):
        shape = [1, signal.shape[1]] + [1] * (signal.ndim - 2)
        self.running_mean = torch.zeros(shape, dtype=signal.dtype, device=signal.device)
        self.running_var = torch.ones(shape, dtype=signal.dtype, device=signal.device)

    def _check_stats(self, signal):
        return (
            self.running_mean is not None
            and self.running_mean.device == signal.device
            and self.running_mean.dtype == signal.dtype
            and self.running_mean.ndim == signal.ndim
            and self.running_mean.shape[1] == signal.shape[1]
        )

    def forward(self, signal):
        if not self._check_stats(signal):
            self._init_stats(signal)
        # Statistics over everything except the channel dimension 1.
        reduce_dims = [0] + list(range(2, signal.ndim))

        if self.training:
            pass
            ## Normalize with the batch mean and the batch variance
            ## (mean of squared deviations), then update running_mean and
            ## running_var in place, without gradients
            ## YOUR CODE HERE
            # -- start --
            mean = signal.mean(dim=reduce_dims, keepdim=True)
            var = (signal - mean).pow(2).mean(dim=reduce_dims, keepdim=True)
            with torch.no_grad():
                self.running_mean.mul_(self.beta).add_(mean, alpha=1 - self.beta)
                self.running_var.mul_(self.beta).add_(var, alpha=1 - self.beta)
            signal = (signal - mean) / (var.sqrt() + self.eps)
            # -- end --
        else:
            pass
            ## Normalize with running_mean and running_var
            ## YOUR CODE HERE
            # -- start --
            signal = (signal - self.running_mean) / (self.running_var.sqrt() + self.eps)
            # -- end --

        return signal


class Residual(torch.nn.Module):
    def __init__(self, module):
        super().__init__()
        ## YOUR CODE HERE
        # -- start --
        self.module = module
        # -- end --

    def forward(self, signal):
        ## YOUR CODE HERE
        # -- start --
        signal = signal + self.module(signal)
        # -- end --

        return signal


class Bottleneck(torch.nn.Module):
    def __init__(
            self, 
            in_channels,
            prenormalization=torch.nn.Identity,
            postnormalization=torch.nn.Identity,
            activation=torch.nn.ReLU,
            compression=1,
            **kwargs):

        super().__init__()
        self.block = torch.nn.Identity()

        ## YOUR CODE HERE
        # -- start --
        self.block = torch.nn.Sequential(
            prenormalization(),
            torch.nn.Linear(in_channels, in_channels // compression, **kwargs),
            postnormalization(),
            activation(),
            prenormalization(),
            torch.nn.Linear(in_channels // compression, in_channels, **kwargs),
            postnormalization(),
        )
        # -- end --

    def forward(self, signal):
        ## YOUR CODE HERE

        # -- start --
        signal = self.block(signal)
        # -- end --
        return signal



class DeepFullyConnectedNet(torch.nn.Module):
    def __init__(
            self,
            block=lambda n_channels: torch.nn.Linear(n_channels, n_channels),
            dim_input=28 * 28,
            dim_embed=128,
            dim_output=10,
            n_blocks=3):
        super().__init__()
        # Placeholder layers: the network runs, wrongly, until they are defined.
        self.encoder = torch.nn.Linear(dim_input, dim_output)
        self.blocks = torch.nn.Identity()
        self.decoder = torch.nn.Identity()

        ## YOUR CODE HERE
        # Define the encoder, the blocks and the decoder

        # -- start --
        self.encoder = torch.nn.Sequential(
            torch.nn.Linear(dim_input, dim_embed)
        )

        blocks = [block(dim_embed) for _ in range(n_blocks)]
        self.blocks = torch.nn.Sequential(*blocks)
        self.decoder = torch.nn.Linear(dim_embed, dim_output)
        # -- end --

    def __forward_kernel(self, signal):
        signal = signal.reshape([signal.shape[0], -1])
        signal = self.encoder(signal)
        signal = self.blocks(signal)
        signal = self.decoder(signal)
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
