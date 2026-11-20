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
        else:
            pass
            ## Normalize with running_mean and running_var
            ## YOUR CODE HERE

        return signal


class Residual(torch.nn.Module):
    def __init__(self, module):
        super().__init__()
        ## YOUR CODE HERE

    def forward(self, signal):
        ## YOUR CODE HERE

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

    def forward(self, signal):
        ## YOUR CODE HERE

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

        # Put the processed result into the batch
        batch['postprocessed'] = {'class': signal}
