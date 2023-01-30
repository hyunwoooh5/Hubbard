import numpy as np
import jax
import jax.numpy as jnp

class HMC:
    def __init__(self, action, init, L=100, dt=0.4):
        self.action = jax.jit(action)
        self._grad = jax.jit(jax.grad(action))
        self.state = init
        self.L = L
        self.dt = dt
        
        self._shape = init.shape
        self._len = np.prod(self._shape, dtype=int)
        # TODO it sure would be nice to allow arbitrary mass matrices
        #self._mass = np.eye(self._len)

        self._steps_proposed = 0
        self._steps_accepted = 0

    def __iter__(self):
        return self

    def __next__(self):
        # Initial position
        x = self.state
        # Initial momentum
        p = np.random.normal(size=self._shape)
        # Initial Hamiltonian
        H0 = np.sum(p**2) / 2 + self.action(x)

        # Leapfrog integration
        for l in range(self.L):
            p -= self.dt/2 * self._grad(x)
            x += self.dt * p
            p -= self.dt/2 * self._grad(x)
            pass

        # Accept/reject
        self._steps_proposed += 1
        H = np.sum(p**2) / 2 + self.action(x)
        if np.random.uniform() < np.exp(H0-H):
            self._steps_accepted += 1
            self.state = x

        return self.state

    def acceptance_rate(self):
        return self._steps_accepted / self._steps_proposed

    def calibrate(self):
        pass

class NUTS:
    def __init__(self):
        pass

    def step(self):
        pass

    def calibrate(self):
        pass
