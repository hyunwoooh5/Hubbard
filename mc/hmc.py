import jax
import jax.numpy as jnp
from functools import partial


class Chain:
    def __init__(self, action, x0, key, L=10, dt=0.3, temperature=1.):
        self.action = jax.jit(lambda y: action(y).real)
        self._grad = jax.jit(jax.grad(lambda y: action(y).real))
        self.x = x0
        self.L = L
        self.dt = dt
        self.temperature = temperature
        self._key = key
        self._recent = [False]

        # @partial(jax.jit, static_argnums=2)
        @jax.jit
        def _propose(key, x, dt):
            kstep, key = jax.random.split(key, 2)
            p = jax.random.normal(kstep, x.shape)

            # initial hamiltonian
            x0 = x
            h0 = jnp.sum(p**2)/2+self.action(x)

            # Leapfrog integration
            for _ in range(self.L):
                p -= dt/2*self._grad(x)
                x += dt * p
                p -= dt/2*self._grad(x)

            # final hamiltonian
            xp = x
            hp = jnp.sum(p**2)/2+self.action(xp)

            return x0, h0, xp, hp

        def _acceptreject(key, temperature, x, h, xp, hp):
            key, kacc = jax.random.split(key, 2)
            hdiff = hp - h

            def accept():
                return xp, True

            def reject():
                return x, False

            acc = jax.random.uniform(kacc) < jnp.exp(-hdiff/temperature)
            x, accepted = jax.lax.cond(acc, accept, reject)

            return key, x, accepted

        self._propose = _propose
        self._acceptreject = jax.jit(_acceptreject)

    def step(self, N=1):
        for _ in range(N):
            x, h, xp, hp = self._propose(self._key, self.x, self.dt)
            self._key, self.x, accepted = self._acceptreject(
                self._key, self.temperature, x, h, xp, hp)
            self._recent.append(accepted)
        self._recent = self._recent[-100:]

    def calibrate(self):
        # Adjust leapfrog steps
        self.step(N=100)
        while self.acceptance_rate() < 0.6 or self.acceptance_rate() > 0.9:
            if self.acceptance_rate() < 0.6:
                self.dt *= 0.98
            if self.acceptance_rate() > 0.9:
                self.dt *= 1.02
            self.step(N=100)

    def acceptance_rate(self):
        return sum(self._recent) / len(self._recent)

    def iter(self, skip=1):
        while True:
            self.step(N=skip)
            yield self.x
