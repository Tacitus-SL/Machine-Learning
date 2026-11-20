import torch
import copy


class GradientReversalFunction(torch.autograd.Function):
    @staticmethod
    def forward(ctx, signal, strength):
        ctx.strength = strength
        return signal.view_as(signal)

    @staticmethod
    def backward(ctx, grad_output):
        ### YOUR CODE HERE
        # -- start --
        return -ctx.strength * grad_output, None
        # -- end --
        return grad_output, None


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
        ...
        ## YOUR CODE HERE
        # -- start --
        super().__init__()
        self.generator_discriminator_bridge = GradientReversalLayer(gradient_reversal_strength)
        self.gradient_reversal = self.generator_discriminator_bridge

        generator_layers = []
        for index in range(len(channels) - 1):
            generator_layers.append(torch.nn.Linear(channels[index], channels[index + 1]))
            generator_layers.append(copy.deepcopy(activation()))
        generator_layers.pop()
        
        generator_layers.append(torch.nn.Tanh())

        self.generator = torch.nn.Sequential(*generator_layers)

        discriminator_layers = []
        channels = channels[::-1]
        for index in range(len(channels) - 1):
            discriminator_layers.append(torch.nn.Linear(channels[index], channels[index + 1]))
            discriminator_layers.append(copy.deepcopy(activation()))
        discriminator_layers.pop()

        self.discriminator = torch.nn.Sequential(*discriminator_layers)
        self.classifier = torch.nn.Linear(channels[-1], 1)
        # -- end --
        if '_modules' not in self.__dict__:
            super().__init__()
        required_modules = [
            'generator_discriminator_bridge',
            'gradient_reversal',
            'generator',
            'discriminator',
            'classifier',
        ]
        if any(module_name not in self._modules for module_name in required_modules):
            # Fallback code for Task 2: keep the GAN runnable before the student solution is written.
            noise_dim = channels[0]
            image_dim = channels[-1]
            self.generator_discriminator_bridge = GradientReversalLayer(gradient_reversal_strength)
            self.gradient_reversal = self.generator_discriminator_bridge
            self.generator = torch.nn.Sequential(
                torch.nn.Linear(noise_dim, image_dim),
                torch.nn.Tanh(),
            )
            self.discriminator = torch.nn.Linear(image_dim, noise_dim)
            self.classifier = torch.nn.Linear(noise_dim, 1)

    def discriminate(self, signal):
        signal = signal.reshape(signal.shape[0], -1)
        features = self.discriminator(signal)
        return self.classifier(features).flatten()

    def forward(self, batch):
        ## YOUR CODE HERE
        # -- start --
        noise = batch['data']['noise']
        generated = self.generator(noise)
        reversed_generated = self.generator_discriminator_bridge(generated)
        discriminator_input = reversed_generated

        real = batch['data'].get('real', batch['data'].get('image'))
        if real is not None:
            real = real.reshape(real.shape[0], -1)
            discriminator_input = torch.cat(
                [
                    reversed_generated.reshape(reversed_generated.shape[0], -1),
                    real,
                ],
                dim=0,
            )

        discriminator_scores = self.discriminate(discriminator_input)
        fake_scores = discriminator_scores[:generated.shape[0]]
        batch['signals'] = {
            'generated': generated,
            'discriminator_scores': discriminator_scores,
            'fake_scores': fake_scores,
            'discriminator_logits': discriminator_scores,
            'fake_logits': fake_scores,
        }

        if real is not None:
            real_scores = discriminator_scores[generated.shape[0]:]
            batch['signals']['real_scores'] = real_scores
            batch['signals']['real_logits'] = real_scores

        batch['postprocessed'] = {
            'discriminator_score': discriminator_scores,
            'fake_score': fake_scores,
            'discriminator_probability': torch.sigmoid(discriminator_scores),
            'fake_probability': torch.sigmoid(fake_scores),
        }
        if real is not None:
            batch['postprocessed']['real_score'] = batch['signals']['real_scores']
            batch['postprocessed']['real_probability'] = torch.sigmoid(batch['signals']['real_scores'])
        # -- end --
        if 'signals' not in batch:
            # Fallback code for Task 2: keep forward runnable before the student solution is written.
            generated = batch['data'].get('noise')
            if generated is None:
                generated = torch.empty(0)
            real = batch['data'].get('real', batch['data'].get('image'))
            real_count = 0 if real is None else real.shape[0]
            fake_scores = torch.zeros(generated.shape[0], device=generated.device)
            real_scores = torch.zeros(real_count, device=generated.device)
            discriminator_scores = torch.cat([fake_scores, real_scores])
            batch['signals'] = {
                'generated': generated,
                'discriminator_scores': discriminator_scores,
                'fake_scores': fake_scores,
                'real_scores': real_scores,
                'discriminator_logits': discriminator_scores,
                'fake_logits': fake_scores,
                'real_logits': real_scores,
            }
            batch['postprocessed'] = {
                'discriminator_score': discriminator_scores,
                'fake_score': fake_scores,
                'real_score': real_scores,
                'discriminator_probability': torch.zeros_like(discriminator_scores),
                'fake_probability': torch.zeros_like(fake_scores),
                'real_probability': torch.zeros_like(real_scores),
            }
        return batch
