# Simple GAN With Gradient Reversal

A generative adversarial network has two trainable parts:

- a generator that maps a random noise vector to an image-shaped vector;
- a discriminator that receives both generated and real images and predicts which inputs are real.

In this exercise we use one shared backpropagation pass instead of alternating separate generator and discriminator steps. The generated samples pass through a `GradientReversalLayer` before the discriminator sees them:

`noise -> generator -> gradient reversal -> discriminator -> classifier`

The gradient-reversal layer does nothing to the forward value. During backpropagation it multiplies the gradient by `-strength`, so the discriminator learns the normal real/fake classification task while the generator receives the opposite signal and learns to make fake images look more real.

The networks use `LeakyReLU(negative_slope=0.5)` by default. The slope is deliberately large: if the discriminator becomes much better than the generator, ordinary ReLU-style activations and saturated classifier logits can leave the generator with tiny useful gradients. A strong leaky branch keeps negative activations informative and helps gradients keep flowing through the discriminator path back toward the generator.
