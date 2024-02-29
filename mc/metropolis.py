import jax
import jax.numpy as jnp


class Chain:
    def __init__(self, action, x0, key, delta=1., temperature=1.):
        self.action = action
        self.x = x0
        self.S = self.action(self.x)
        self.delta = delta
        self.temperature = temperature
        self._key = key
        self._recent = [False]

        def _propose(key, x, delta):
            kstep, key = jax.random.split(key, 2)
            xp = x + delta*jax.random.normal(kstep, x.shape)

            return xp

        def _acceptreject(key, temperature, x, S, xp, Sp):
            key, kacc = jax.random.split(key, 2)
            Sdiff = Sp - S

            def accept():
                return xp, Sp, True

            def reject():
                return x, S, False

            acc = jax.random.uniform(kacc) < jnp.exp(-Sdiff/temperature)
            x, S, accepted = jax.lax.cond(acc, accept, reject)

            return key, x, S, accepted

        self._propose = jax.jit(_propose)
        self._acceptreject = jax.jit(_acceptreject)

    def _action(self, key, x, delta):
        xp = self._propose(key, x, delta)
        Sp = self.action(xp).real
        return xp, Sp

    def step(self, N=1):
        self.S = self.action(self.x).real
        for _ in range(N):
            xp, Sp = self._action(self._key, self.x, self.delta)
            self._key, self.x, self.S, accepted = self._acceptreject(
                self._key, self.temperature, self.x, self.S, xp, Sp)
            self._recent.append(accepted)
        self._recent = self._recent[-100:]

    def calibrate(self):
        # Adjust delta.
        self.step(N=100)
        while self.acceptance_rate() < 0.3 or self.acceptance_rate() > 0.55:
            if self.acceptance_rate() < 0.3:
                self.delta *= 0.98
            if self.acceptance_rate() > 0.55:
                self.delta *= 1.02
            self.step(N=100)

    def acceptance_rate(self):
        return sum(self._recent) / len(self._recent)

    def iter(self, skip=1):
        while True:
            self.step(N=skip)
            yield self.x
