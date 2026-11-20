"""Optimizers written from their update formulas.

Every optimizer here keeps a list of parameters and, for each parameter,
a small ``state`` dictionary (momentum, running sums, step count). The
loop over parameters lives in ``Optimizer.step``; each optimizer only says
how one parameter changes, in ``update``.
"""
import contextlib

import torch


class Optimizer:
    """Loop over parameters and call ``update`` for each one with a gradient.

    Parameters
    ----------
    params : iterable of torch.Tensor
        Tensors to optimize, e.g. ``model.parameters()``.
    lr : float
        Learning rate.
    """

    def __init__(self, params, lr):
        self.params = list(params)
        self.lr = lr
        self.state = {param: {} for param in self.params}

    def zero_grad(self):
        for param in self.params:
            param.grad = None

    @torch.no_grad()
    def step(self):
        for param in self.params:
            if param.grad is not None:
                self.update(param, param.grad, self.state[param])

    def update(self, param, grad, state):
        """Change ``param`` in place, using ``grad`` and its own ``state``."""
        raise NotImplementedError


class GradientDescent(Optimizer):
    """``param <- param - lr * grad``."""

    def update(self, param, grad, state):
        pass
        ## YOUR CODE HERE
        # -- start --
        param -= self.lr * grad
        # -- end --


class Momentum(Optimizer):
    """Gradient descent along a running sum of gradients.

    ``velocity <- beta * velocity + grad``, then
    ``param <- param - lr * velocity``.
    """

    def __init__(self, params, lr, beta=0.9):
        super().__init__(params, lr)
        self.beta = beta

    def update(self, param, grad, state):
        if not state:
            state['velocity'] = torch.zeros_like(param)
        velocity = state['velocity']
        ## YOUR CODE HERE
        # -- start --
        velocity.mul_(self.beta).add_(grad)
        param -= self.lr * velocity
        # -- end --


class AdaGrad(Optimizer):
    """Per-coordinate steps that shrink where gradients were large.

    ``squares <- squares + grad ** 2``, then
    ``param <- param - lr * grad / (sqrt(squares) + eps)``.
    """

    def __init__(self, params, lr, eps=1.0e-10):
        super().__init__(params, lr)
        self.eps = eps

    def update(self, param, grad, state):
        if not state:
            state['squares'] = torch.zeros_like(param)
        squares = state['squares']
        ## YOUR CODE HERE
        # -- start --
        squares += grad ** 2
        param -= self.lr * grad / (squares.sqrt() + self.eps)
        # -- end --


class Adam(Optimizer):
    """Momentum divided by a running gradient scale, with bias correction.

    ``mean <- beta1 * mean + (1 - beta1) * grad``,
    ``square <- beta2 * square + (1 - beta2) * grad ** 2``,
    ``param <- param - lr * mean_hat / (sqrt(square_hat) + eps)``, where
    ``mean_hat = mean / (1 - beta1 ** step)`` and
    ``square_hat = square / (1 - beta2 ** step)``.
    """

    def __init__(self, params, lr=1.0e-3, betas=(0.9, 0.999), eps=1.0e-8):
        super().__init__(params, lr)
        self.beta1, self.beta2 = betas
        self.eps = eps

    def update(self, param, grad, state):
        if not state:
            state['step'] = 0
            state['mean'] = torch.zeros_like(param)
            state['square'] = torch.zeros_like(param)
        state['step'] += 1
        step, mean, square = state['step'], state['mean'], state['square']
        ## YOUR CODE HERE
        # -- start --
        mean.mul_(self.beta1).add_((1 - self.beta1) * grad)
        square.mul_(self.beta2).add_((1 - self.beta2) * grad ** 2)
        mean_hat = mean / (1 - self.beta1 ** step)
        square_hat = square / (1 - self.beta2 ** step)
        param -= self.lr * mean_hat / (square_hat.sqrt() + self.eps)
        # -- end --


class Lion(Optimizer):
    """Steps of the same size in every coordinate: only the sign is used.

    ``param <- param - lr * sign(beta1 * mean + (1 - beta1) * grad)``, then
    ``mean <- beta2 * mean + (1 - beta2) * grad``.
    """

    def __init__(self, params, lr=1.0e-4, betas=(0.9, 0.99)):
        super().__init__(params, lr)
        self.beta1, self.beta2 = betas

    def update(self, param, grad, state):
        if not state:
            state['mean'] = torch.zeros_like(param)
        mean = state['mean']
        ## YOUR CODE HERE
        # -- start --
        param -= self.lr * torch.sign(self.beta1 * mean + (1 - self.beta1) * grad)
        mean.mul_(self.beta2).add_((1 - self.beta2) * grad)
        # -- end --


def orthogonalize(matrix, steps=5, eps=1.0e-7):
    """Push all singular values of a 2-D ``matrix`` close to 1.

    Quintic Newton-Schulz iteration from Muon
    (https://github.com/KellerJordan/Muon): ``x <- a x + (b G + c G^2) x``
    with ``G = x x^T``. The result keeps the singular vectors of
    ``matrix``; its singular values end up roughly in [0.7, 1.2].
    """
    a, b, c = 3.4445, -4.7750, 2.0315
    x = matrix / matrix.norm().clamp(min=eps)
    is_tall = x.shape[0] > x.shape[1]
    if is_tall:
        x = x.T
    for _ in range(steps):
        gram = x @ x.T
        x = a * x + (b * gram + c * gram @ gram) @ x
    return x.T if is_tall else x


class Muon(Optimizer):
    """Momentum whose step is orthogonalized, for 2-D weight matrices.

    ``mean <- beta * mean + (1 - beta) * grad``, then
    ``param <- param - lr * scale * orthogonalize(mean)`` with
    ``scale = sqrt(max(1, rows / columns))``. Parameters that are not
    matrices (biases) are updated by ``Adam`` with ``adam_lr`` instead.
    """

    def __init__(self, params, lr=0.02, beta=0.95, adam_lr=1.0e-3):
        super().__init__(params, lr)
        self.beta = beta
        self.adam = Adam([param for param in self.params if param.ndim != 2],
                         lr=adam_lr)

    def update(self, param, grad, state):
        if param.ndim != 2:
            self.adam.update(param, grad, self.adam.state[param])
            return
        if not state:
            state['mean'] = torch.zeros_like(param)
        mean = state['mean']
        scale = max(1.0, param.shape[0] / param.shape[1]) ** 0.5
        ## YOUR CODE HERE
        # -- start --
        mean.mul_(self.beta).add_((1 - self.beta) * grad)
        param -= self.lr * scale * orthogonalize(mean)
        # -- end --


class Lookahead:
    """Every ``k`` steps, move slow weights towards the fast ones.

    The wrapped ``optimizer`` moves the parameters (fast weights). Every
    ``k`` steps: ``slow <- slow + alpha * (fast - slow)``, then
    ``fast <- slow``.
    """

    def __init__(self, optimizer, k=5, alpha=0.5):
        self.optimizer = optimizer
        self.k = k
        self.alpha = alpha
        self.slow = {param: param.detach().clone() for param in optimizer.params}
        self.step_count = 0

    @property
    def lr(self):
        return self.optimizer.lr

    @lr.setter
    def lr(self, value):
        self.optimizer.lr = value

    def zero_grad(self):
        self.optimizer.zero_grad()

    @torch.no_grad()
    def step(self):
        self.optimizer.step()
        self.step_count += 1
        if self.step_count % self.k != 0:
            return
        for fast, slow in self.slow.items():
            ## YOUR CODE HERE
            # -- start --
            slow += self.alpha * (fast - slow)
            # -- end --
            fast.copy_(slow)


class WeightAverage:
    """Exponential moving average of the weights, kept next to training.

    After every step of the wrapped ``optimizer``:
    ``average <- decay * average + (1 - decay) * param``. Evaluate inside
    ``with weight_average.averaged_weights():`` to use the averages.
    """

    def __init__(self, optimizer, decay=0.9):
        self.optimizer = optimizer
        self.decay = decay
        self.averages = {param: param.detach().clone()
                         for param in optimizer.params}

    @property
    def lr(self):
        return self.optimizer.lr

    @lr.setter
    def lr(self, value):
        self.optimizer.lr = value

    def zero_grad(self):
        self.optimizer.zero_grad()

    @torch.no_grad()
    def step(self):
        self.optimizer.step()
        for param, average in self.averages.items():
            pass
            ## YOUR CODE HERE
            # -- start --
            average.mul_(self.decay).add_((1 - self.decay) * param)
            # -- end --

    @contextlib.contextmanager
    def averaged_weights(self):
        """Swap the averages into the parameters for the ``with`` block."""
        with torch.no_grad():
            saved = {param: param.detach().clone() for param in self.averages}
            for param, average in self.averages.items():
                param.copy_(average)
        try:
            yield
        finally:
            with torch.no_grad():
                for param, value in saved.items():
                    param.copy_(value)


class ReduceLROnPlateau:
    """Multiply the learning rate by ``factor`` when the loss stops falling.

    Call ``step(loss)`` once per epoch with the validation loss. A loss
    below ``best * (1 - threshold)`` is an improvement. After more than
    ``patience`` epochs without one: ``lr <- factor * lr``, and the count
    of bad epochs starts again from zero.
    """

    # train_model passes the validation loss to step() when this is set.
    steps_on_validation_loss = True

    def __init__(self, optimizer, factor=0.1, patience=2, threshold=1.0e-4):
        self.optimizer = optimizer
        self.factor = factor
        self.patience = patience
        self.threshold = threshold
        self.best = float('inf')
        self.bad_epochs = 0

    def step(self, loss):
        if loss < self.best * (1 - self.threshold):
            self.best = loss
            self.bad_epochs = 0
        else:
            self.bad_epochs += 1
        if self.bad_epochs > self.patience:
            ## YOUR CODE HERE
            # -- start --
            self.optimizer.lr *= self.factor
            # -- end --
            self.bad_epochs = 0


def check_steps(make_optimizer, n_steps=3):
    """Run ``n_steps`` of an optimizer on a fixed problem; return the weights.

    The problem is a fixed least-squares fit of a 3 x 2 matrix and a bias,
    so any two correct implementations of the same formula give the same
    numbers.

    Parameters
    ----------
    make_optimizer : callable
        ``make_optimizer(params)`` returns an optimizer for the list
        ``params``.
    n_steps : int
        Number of optimizer steps.

    Returns
    -------
    dict
        ``{'weight': ..., 'bias': ...}`` numpy arrays after the last step;
        for a ``WeightAverage`` also ``'average_weight'`` and
        ``'average_bias'``.
    """
    generator = torch.Generator().manual_seed(0)
    weight = torch.randn(3, 2, generator=generator).requires_grad_()
    bias = torch.randn(3, generator=generator).requires_grad_()
    inputs = torch.randn(8, 2, generator=generator)
    targets = torch.randn(8, 3, generator=generator)
    optimizer = make_optimizer([weight, bias])
    for _ in range(n_steps):
        optimizer.zero_grad()
        loss = ((inputs @ weight.T + bias - targets) ** 2).mean()
        loss.backward()
        optimizer.step()
    result = {'weight': weight.detach().numpy().copy(),
              'bias': bias.detach().numpy().copy()}
    if isinstance(optimizer, WeightAverage):
        result['average_weight'] = optimizer.averages[weight].numpy().copy()
        result['average_bias'] = optimizer.averages[bias].numpy().copy()
    return result
