from dataclasses import dataclass
from typing import Tuple
from itertools import product
from jax.scipy.linalg import expm
from functools import partial

import jax
import jax.numpy as jnp
import numpy as np

# Don't print annoying CPU warning.
jax.config.update('jax_platform_name', 'cpu')


@dataclass  # 2D model, LxL lattice
class Lattice:
    #    geom: Tuple[int]
    L: int
    nt: int

    def __post_init__(self):
        #       self.D = len(self.geom)
        self.V = self.L**2
        self.dof = self.V * self.nt
        self.periodic_contour = True

    def idx(self, t, x0, x1):
        return x1 + self.L*x0 + self.L*self.L*t

    def idx1(self, x0, x1):
        return x1 + self.L*x0

    def sites(self):
        # Return a list of all sites.
        return jnp.indices((self.nt, self.V))

    def coords(self, i):
        t = i//self.L
        x = i % self.L
        return t, x


def hopping(l):
    hop = jnp.zeros((l**2, l**2))
    a = Lattice(l, 1)

    for x1, y1, x2, y2 in product(range(l), range(l), range(l), range(l)):
        if (x1 == x2 and (y1 == (y2+1) % l or y1 == (y2-1+l) % l)):
            hop = hop.at[a.idx1(x1, y1), a.idx1(x2, y2)].add(1.0)
        if (y1 == y2 and (x1 == (x2+1) % l or x1 == (x2-1+l) % l)):
            hop = hop.at[a.idx1(x1, y1), a.idx1(x2, y2)].add(1.0)

    return hop


def exp_h1(l, kappa, mu):
    h1 = kappa * hopping(l)
    for i in range(l**2):
        h1 = h1.at[i, i].add(mu)
    h1 = expm(h1)
    return h1


def exp_h2(l, kappa, mu):
    h2 = kappa * hopping(l)
    for i in range(l**2):
        h2 = h2.at[i, i].add(-mu)
    h2 = expm(h2)
    return h2
