import jax.numpy as jnp

class WrapHbar:
    """
    A wrapper that modifies hbar.
    """

    def __init__(self, model, hbar):
        self.model = model
        self.hbar = hbar

    def action(self, x):
        return self.model.action(x)/self.hbar

    def observe(self, x):
        return self.model.observe(x)

