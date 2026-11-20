"""A small positional-encoding NeRF with volume rendering."""

import torch
from torch import nn
from torch.nn import functional as F


class TinyNeRF(nn.Module):
    """Predict density and RGB at points along each ray and render the ray.

    ``forward`` follows the course batch protocol: it reads
    ``batch['data']['ray_origin']`` and ``['ray_direction']`` (``[batch, 3]``)
    and writes ``batch['signals']['rgb']`` (``[batch, 3]``), ``['weights']``
    (``[batch, n_samples]``) and ``['depth']`` (``[batch]``).
    """

    def __init__(self, hidden_size=128, n_frequencies=6, n_samples=32,
                 near=2.0, far=6.0):
        super().__init__()
        self.hidden_size = int(hidden_size)
        self.n_frequencies = int(n_frequencies)
        self.n_samples = int(n_samples)
        self.near, self.far = float(near), float(far)
        encoded_size = 3 * (1 + 2 * self.n_frequencies)
        self.field = nn.Identity()
        self.density_head = nn.Linear(encoded_size, 1)
        self.color_head = nn.Linear(encoded_size, 3)
        ## Define self.field (Linear -> ReLU -> Linear -> ReLU, hidden_size wide)
        ## and the two heads that read its output
        ## YOUR CODE HERE
        # -- start --
        self.field = nn.Sequential(
            nn.Linear(encoded_size, self.hidden_size),
            nn.ReLU(),
            nn.Linear(self.hidden_size, self.hidden_size),
            nn.ReLU(),
        )
        self.density_head = nn.Linear(self.hidden_size, 1)
        self.color_head = nn.Linear(self.hidden_size, 3)
        # -- end --

    def positional_encoding(self, coords):
        """Append sine and cosine features at powers-of-two frequencies."""
        encoded = [coords]
        ## Append sin(2^k pi coords) and cos(2^k pi coords) for k = 0, ..., n_frequencies - 1
        ## YOUR CODE HERE
        # -- start --
        for frequency_index in range(self.n_frequencies):
            frequency = (2.0 ** frequency_index) * torch.pi
            encoded.append(torch.sin(frequency * coords))
            encoded.append(torch.cos(frequency * coords))
        # -- end --
        if len(encoded) == 1:
            encoded.extend(torch.zeros_like(coords) for _ in range(2 * self.n_frequencies))
        return torch.cat(encoded, dim=-1)

    def query_field(self, coords):
        """Return non-negative density and RGB in [0, 1]."""
        features = self.field(self.positional_encoding(coords))
        density = F.softplus(self.density_head(features)).squeeze(-1)
        color = torch.sigmoid(self.color_head(features))
        return density, color

    def sample_depths(self, n_rays):
        """Depths of the points on each ray and the step between them.

        The segment from ``near`` to ``far`` is cut into ``n_samples`` equal
        bins. In training, every ray takes a random point in each bin, so
        over many steps the network sees the whole ray; in evaluation, the
        middle of each bin.
        """
        step = (self.far - self.near) / self.n_samples
        bin_starts = self.near + step * torch.arange(self.n_samples, dtype=torch.float32)
        if self.training:
            offsets = torch.rand(n_rays, self.n_samples)
        else:
            offsets = torch.full((n_rays, self.n_samples), 0.5)
        depths = bin_starts + offsets * step
        return depths, torch.full_like(depths, step)

    def render(self, samples, deltas):
        """Alpha-composite density and color along each ray."""
        batch_size, n_samples, _ = samples.shape
        density, color = self.query_field(samples.reshape(-1, 3))
        density = density.reshape(batch_size, n_samples)
        color = color.reshape(batch_size, n_samples, 3)
        weights = None
        rgb = None
        ## Opacity alpha = 1 - exp(-density * delta); the weight of a sample is
        ## its alpha times the transparency of all the samples before it;
        ## rgb = sum of weight * color along the ray
        ## YOUR CODE HERE
        # -- start --
        alpha = 1.0 - torch.exp(-density * deltas)
        transmittance = torch.cumprod(
            torch.cat((torch.ones_like(alpha[:, :1]),
                       1.0 - alpha[:, :-1] + 1.0e-10), dim=1),
            dim=1,
        )
        weights = transmittance * alpha
        rgb = (weights[..., None] * color).sum(dim=1)
        # -- end --
        if weights is None:
            weights = torch.zeros_like(density)
            rgb = color.sum(dim=(1, 2), keepdim=False)[:, None] * 0
            rgb = rgb.expand(-1, 3)
        return rgb, weights, density, color

    def forward(self, batch):
        origins = batch['data']['ray_origin']
        directions = batch['data']['ray_direction']
        depths, deltas = self.sample_depths(len(origins))
        samples = origins[:, None, :] + depths[..., None] * directions[:, None, :]
        rgb, weights, density, color = self.render(samples, deltas)
        batch['signals'] = {
            'rgb': rgb,
            'weights': weights,
            'depth': (weights * depths).sum(dim=1),
        }
        batch['postprocessed'] = {'rgb': rgb.clamp(0.0, 1.0)}
        return batch
