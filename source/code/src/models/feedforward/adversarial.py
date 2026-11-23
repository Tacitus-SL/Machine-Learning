import torch
import copy


class GradientReversalFunction(torch.autograd.Function):
    @staticmethod
    def forward(ctx, signal, strength):
        ctx.strength = strength
        return signal.view_as(signal)

    @staticmethod
    def backward(ctx, grad_output):
        return -ctx.strength * grad_output, None


class GradientReversalLayer(torch.nn.Module):
    def __init__(self, strength=1.0):
        super().__init__()
        self.strength = float(strength)

    def forward(self, signal):
        return GradientReversalFunction.apply(signal, self.strength)


class GAN(torch.nn.Module):
    def __init__(
            self,
            channels,
            gradient_reversal_strength=1.0,
            activation=lambda: torch.nn.LeakyReLU(negative_slope=0.5)
    ):
        super().__init__()
        generator_layers = []
        for i in range(len(channels) - 1):
            generator_layers.append(torch.nn.Linear(channels[i], channels[i + 1]))
            if i < len(channels) - 2:
                generator_layers.append(activation())
            else:
                generator_layers.append(torch.nn.Tanh())
        self.generator = torch.nn.Sequential(*generator_layers)

        self.generator_discriminator_bridge = GradientReversalLayer(gradient_reversal_strength)
        self.gradient_reversal = self.generator_discriminator_bridge

        discriminator_layers = []
        for i in range(len(channels) - 1, 0, -1):
            discriminator_layers.append(torch.nn.Linear(channels[i], channels[i - 1]))
            if i > 1:
                discriminator_layers.append(activation())
        self.discriminator = torch.nn.Sequential(*discriminator_layers)

        self.classifier = torch.nn.Linear(channels[0], 1)

    def discriminate(self, signal):
        signal = signal.reshape(signal.shape[0], -1)
        features = self.discriminator(signal)
        return self.classifier(features).flatten()

    def forward(self, batch):
        data = batch['data']
        noise = data['noise'].reshape(data['noise'].shape[0], -1)
        fake_count = noise.shape[0]

        generated = self.generator(noise)
        reversed_generated = self.generator_discriminator_bridge(generated)

        real_raw = data.get('real', data.get('image'))

        signals = {'generated': generated}
        postprocessed = {}

        if real_raw is not None:
            real_flat = real_raw.reshape(real_raw.shape[0], -1)
            disc_input = torch.cat([reversed_generated, real_flat], dim=0)
            logits = self.discriminate(disc_input)

            fake_logits = logits[:fake_count]
            real_logits = logits[fake_count:]

            signals['discriminator_logits'] = logits
            signals['fake_logits'] = fake_logits
            signals['real_logits'] = real_logits

            signals['discriminator_scores'] = logits
            signals['fake_scores'] = fake_logits
            signals['real_scores'] = real_logits

            disc_probs = torch.sigmoid(logits)
            fake_probs = torch.sigmoid(fake_logits)
            real_probs = torch.sigmoid(real_logits)

            postprocessed['discriminator_score'] = logits
            postprocessed['fake_score'] = fake_logits
            postprocessed['real_score'] = real_logits
            postprocessed['discriminator_probability'] = disc_probs
            postprocessed['fake_probability'] = fake_probs
            postprocessed['real_probability'] = real_probs
        else:
            fake_logits = self.discriminate(reversed_generated)

            signals['discriminator_logits'] = fake_logits
            signals['fake_logits'] = fake_logits
            signals['discriminator_scores'] = fake_logits
            signals['fake_scores'] = fake_logits

            fake_probs = torch.sigmoid(fake_logits)

            postprocessed['discriminator_score'] = fake_logits
            postprocessed['fake_score'] = fake_logits
            postprocessed['discriminator_probability'] = fake_probs
            postprocessed['fake_probability'] = fake_probs

        batch['signals'] = signals
        batch['postprocessed'] = postprocessed
        return batch
