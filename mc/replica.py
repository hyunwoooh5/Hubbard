import jax
import jax.numpy as jnp
import numpy as np

from mc import metropolis

class ReplicaExchange:
    def __init__(self, action, x0, key, Chain=metropolis.Chain, max_hbar=10., Nreplicas=30):
        self.Nreplicas = Nreplicas

        loghbars = np.linspace(0,np.log(max_hbar),Nreplicas)
        hbars = np.exp(loghbars)
        key, self.key = jax.random.split(key, 2)
        keys = jax.random.split(key, Nreplicas)
        self.replicas = [Chain(action, x0, key, temperature=hbar) for hbar,key in zip(hbars,keys)]

        def _propose(key):
            kswap, key = jax.random.split(key, 2)
            kswapi, kswapj = jax.random.split(kswap, 2)
            i=jax.random.randint(kswapi, shape=(), minval=0, maxval=self.Nreplicas)
            j=jax.random.randint(kswapj, shape=(), minval=0, maxval=self.Nreplicas)
            
            return i, j, key

        def _propose2(key):
            kacc, key = jax.random.split(key, 2)
            x = jax.random.uniform(kacc)
            return x

        self._propose=jax.jit(_propose)
        self._propose2=jax.jit(_propose2)

    def step(self, N=1):
        for n in range(N):
            for replica in self.replicas:
                replica.step()
            self.exchange()
        self.x = self.replicas[0].x

    def exchange(self):
        if self.Nreplicas == 1:
            return
        for _ in range(self.Nreplicas):
            i, j, self.key = self._propose(self.key)
            if i == j:
                continue
            hbari, hbarj = self.replicas[i].temperature, self.replicas[j].temperature
            Si, Sj = self.replicas[i].S, self.replicas[j].S
            S = Si/hbari + Sj/hbarj
            Sp = Sj/hbari + Si/hbarj
            Sdiff = Sp - S
            if Sdiff < 0 or self._propose2(self.key) < np.exp(-Sdiff):
                self._swap(i,j)

    def _swap(self, i, j):
        temperaturei, temperaturej = self.replicas[i].temperature, self.replicas[j].temperature
        self.replicas[i].temperature, self.replicas[j].temperature = temperaturej, temperaturei
        self.replicas[i], self.replicas[j] = self.replicas[j], self.replicas[i]

    def calibrate(self):
        for replica in self.replicas:
            replica.calibrate()

    def iter(self, skip=1):
        while True:
            self.step(N=skip)
            yield self.replicas[0].x

    def acceptance_rate(self):
        # TODO pass on information about exchanges
        return np.mean([r.acceptance_rate() for r in self.replicas])

