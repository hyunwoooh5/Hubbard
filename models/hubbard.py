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


@dataclass
class Hopping:
    lattice: Lattice
    kappa: float
    mu: float

    def hopping(self):
        size = self.lattice.L
        hop = jnp.zeros((size**2, size**2))
        a = Lattice(self.lattice.L, 1)

        for x1, y1, x2, y2 in product(range(size), range(size), range(size), range(size)):
            if (x1 == x2 and (y1 == (y2+1) % size or y1 == (y2-1+size) % size)):
                hop = hop.at[a.idx1(x1, y1), a.idx1(x2, y2)].add(1.0)
            if (y1 == y2 and (x1 == (x2+1) % size or x1 == (x2-1+size) % size)):
                hop = hop.at[a.idx1(x1, y1), a.idx1(x2, y2)].add(1.0)

        return hop

    def exp_h1(self):
        h1 = self.kappa * self.hopping()
        for i in range(self.lattice.L**2):
            h1 = h1.at[i, i].add(self.mu)
        h1 = expm(h1)
        return h1

    def exp_h2(self):
        h2 = self.kappa * self.hopping()
        for i in range(self.lattice.L**2):
            h2 = h2.at[i, i].add(-self.mu)
        h2 = expm(h2)
        return h2


@dataclass
class ImprovedModel:
    lattice: Lattice
    beta: float
    Kappa: float
    U: float
    Mu: float
    dt: float

    def __post_init__(self):
        self.kappa = self.Kappa * self.dt
        self.u = self.U * self.dt
        self.mu = self.Mu * self.dt
        self.Hopping = Hopping(self.lattice, self.kappa, self.mu)

        self.hopping = self.Hopping.hopping()
        self.h1 = self.Hopping.exp_h1()
        self.h2 = self.Hopping.exp_h2()

    def Hubbard1_old(self, A):
        idx = self.lattice.idx
        # A = A.reshape((self.lattice.nt, self.lattice.V))
        fer_mat1 = jnp.eye(self.lattice.V) + 0j

        for i in range(self.lattice.nt):
            for x in range(self.lattice.V):
                for y in range(self.lattice.V):
                    fer_mat1 = fer_mat1.at[x, y].multiply(jnp.exp(1j *
                                                                  jnp.sin(A[i * self.lattice.V + x])))

            fer_mat1 = self.h1 @ fer_mat1

        return jnp.eye(self.lattice.V) + fer_mat1

    def Hubbard2_old(self, A):
        idx = self.lattice.idx
        # A = A.reshape((self.lattice.nt, self.lattice.V))
        fer_mat2 = jnp.eye(self.lattice.V) + 0j

        for i in range(self.lattice.nt):
            for x in range(self.lattice.V):
                for y in range(self.lattice.V):
                    fer_mat2 = fer_mat2.at[x, y].multiply(jnp.exp(-1j *
                                                                  jnp.sin(A[i * self.lattice.V + x])))

            fer_mat2 = self.h2 @ fer_mat2

        return jnp.eye(self.lattice.V) + fer_mat2

    def Hubbard1_new(self, A):
        idx = self.lattice.idx
#        A = A.reshape((self.lattice.nt, self.lattice.L, self.lattice.L))
        fer_mat1 = jnp.eye(self.lattice.V) + 0j

        def update_at_tx(t, x, H):
            H = H.at[x, x].add(
                jnp.exp(1j * jnp.sin(A[t * self.lattice.V + x])))
            return H

        def update_at_t(t):
            some = partial(update_at_tx, t)
            temp_mat = jnp.zeros((self.lattice.V, self.lattice.V)) + 0j
            temp_mat = jax.lax.fori_loop(0, self.lattice.V, some, temp_mat)
            return self.h1 @ temp_mat

        def multi(t, fer_mat):
            return update_at_t(t) @ fer_mat

        fer_mat1 = jax.lax.fori_loop(0, self.lattice.nt, multi, fer_mat1)

        return jnp.eye(self.lattice.V) + fer_mat1

    def Hubbard2_new(self, A):
        idx = self.lattice.idx
#        A = A.reshape((self.lattice.nt, self.lattice.L, self.lattice.L))
        fer_mat2 = jnp.eye(self.lattice.V) + 0j

        def update_at_tx(t, x, H):
            H = H.at[x, x].add(
                jnp.exp(-1j * jnp.sin(A[t * self.lattice.V + x])))
            return H

        def update_at_t(t):
            some = partial(update_at_tx, t)
            temp_mat = jnp.zeros((self.lattice.V, self.lattice.V)) + 0j
            temp_mat = jax.lax.fori_loop(0, self.lattice.V, some, temp_mat)
            return self.h2 @ temp_mat

        def multi(t, fer_mat):
            return update_at_t(t) @ fer_mat

        fer_mat2 = jax.lax.fori_loop(0, self.lattice.nt, multi, fer_mat2)

        return jnp.eye(self.lattice.V) + fer_mat2

    def Hubbard1(self, A):
        return self.Hubbard1_new(A)

    def Hubbard2(self, A):
        return self.Hubbard2_new(A)

    def action(self, A):
        s, logdet = jnp.linalg.slogdet(self.Hubbard1(A) @ self.Hubbard2(A))
        return -self.beta * jnp.sum(jnp.cos(A)) - jnp.log(s) - logdet

    def density(self, A):
        idx = self.lattice.idx
        # A = A.reshape((self.lattice.nt, self.lattice.V))

        fer_mat1_inv = jnp.linalg.inv(self.Hubbard1(A))
        fer_mat2_inv = jnp.linalg.inv(self.Hubbard2(A))

        return jnp.trace(fer_mat2_inv - fer_mat1_inv) / self.lattice.V

    def doubleoccupancy(self, A):
        d = 0 + 0j
        fer_mat1_inv = jnp.linalg.inv(self.Hubbard1(A))
        fer_mat2_inv = jnp.linalg.inv(self.Hubbard2(A))

        def do(i, d):
            return d + fer_mat2_inv[i, i] * (1.0 - fer_mat1_inv[i, i])

        d = jax.lax.fori_loop(0, self.lattice.V, do, d) / self.lattice.V

        return d

    def staggered_magnetization(self, A):
        m = 0 + 0j
        fer_mat1_inv = jnp.linalg.inv(self.Hubbard1(A))
        fer_mat2_inv = jnp.linalg.inv(self.Hubbard2(A))

        idx = self.lattice.idx1
        size = self.lattice.L
        A = A.reshape((self.lattice.nt, size, size))
        x = jnp.array(range(size))

        def update_at(x1, y1, x2, y2):
            a = idx(x1, y1)
            b = idx(x2, y2)

            m = - (-1.0)**(x1 + y1 + x2 + y2) * fer_mat1_inv[a, b] * \
                fer_mat1_inv[b, a] + fer_mat2_inv[a, b] * fer_mat2_inv[b, a]

            if (x1 == x2 and y1 == y2):
                m += fer_mat1_inv[a, a] * fer_mat2_inv[a, a]

            return m

        m = jax.lax.fori_loop(0, size, lambda i, vx: vx + jax.lax.fori_loop(0, size,
                                                                            lambda j, vy: vy + jax.lax.fori_loop(0, size,
                                                                                                                 lambda k, vz: vz + jax.lax.fori_loop(0, size,
                                                                                                                                                      lambda l, vw: vw + update_at(x[i], x[j], x[k], x[l]), m), m), m), m)

        ''' for representation
        for x1, y1, x2, y2 in product(range(size), range(size), range(size), range(size)):
            a = idx(x1, y1)
            b = idx(x2, y2)

            m -= (-1.0)**(x1 + y1 + x2 + y2) * fer_mat1_inv(a, b) * \
            fer_mat1_inv(b, a) + fer_mat2_inv(a, b) * fer_mat2_inv(b, a)

            if (x1 == x2 and y1 == y2):
                m += fer_mat1_inv(a, a) * fer_mat2_inv(a, a)        
        '''
        return m

    def magnetization(self, A):
        m = 0 + 0j
        fer_mat1_inv = jnp.linalg.inv(self.Hubbard1(A))
        fer_mat2_inv = jnp.linalg.inv(self.Hubbard2(A))

        idx = self.lattice.idx1
        size = self.lattice.L
        A = A.reshape((self.lattice.nt, size, size))
        x = jnp.array(range(size))

        def update_at(x1, y1, x2, y2):
            a = idx(x1, y1)
            b = idx(x2, y2)

            m = - fer_mat1_inv[a, b] * fer_mat1_inv[b, a] + \
                fer_mat2_inv[a, b] * fer_mat2_inv[b, a]

            if (x1 == x2 and y1 == y2):
                m += fer_mat1_inv[a, a] * fer_mat2_inv[a, a]

            return m

        m = jax.lax.fori_loop(0, size, lambda i, vx: vx + jax.lax.fori_loop(0, size,
                                                                            lambda j, vy: vy + jax.lax.fori_loop(0, size,
                                                                                                                 lambda k, vz: vz + jax.lax.fori_loop(0, size,
                                                                                                                                                      lambda l, vw: vw + update_at(x[i], x[j], x[k], x[l]), m), m), m), m)

        ''' for representation
        for x1, y1, x2, y2 in product(range(size), range(size), range(size), range(size)):
            a = idx(x1, y1)
            b = idx(x2, y2)

            m -= fer_mat1_inv(a, b) * fer_mat1_inv(b, a) + \
            fer_mat2_inv(a, b) * fer_mat2_inv(b, a)

            if (x1 == x2 and y1 == y2):
                m += fer_mat1_inv(a, a) * fer_mat2_inv(a, a)
        '''
        return m

    def n1(self, A):
        fer_mat1_inv = jnp.linalg.inv(self.Hubbard1(A))

        return jnp.trace(fer_mat1_inv) / self.lattice.V

    def n2(self, A):
        fer_mat2_inv = jnp.linalg.inv(self.Hubbard2(A))

        return jnp.trace(fer_mat2_inv) / self.lattice.V

    def hamiltonian(self, A):
        h = 0 + 0j
        fer_mat1_inv = jnp.linalg.inv(self.Hubbard1(A))
        fer_mat2_inv = jnp.linalg.inv(self.Hubbard2(A))

        idx = self.lattice.idx1
        size = self.lattice.L
        A = A.reshape((self.lattice.nt, size, size))
        x = jnp.array(range(size))

        def update_at(x1, y1, x2, y2):
            a = idx(x1, y1)
            b = idx(x2, y2)

            h = 0.5 * self.kappa * \
                self.hopping[a, b] * (fer_mat1_inv[a, b] + fer_mat2_inv[a, b])

            if (x1 == x2 and y1 == y2):
                m += fer_mat1_inv[a, a] * fer_mat2_inv[a, a]

            return h

        h = jax.lax.fori_loop(0, size, lambda i, vx: vx + jax.lax.fori_loop(0, size,
                                                                            lambda j, vy: vy + jax.lax.fori_loop(0, size,
                                                                                                                 lambda k, vz: vz + jax.lax.fori_loop(0, size,
                                                                                                                                                      lambda l, vw: vw + update_at(x[i], x[j], x[k], x[l]), h), h), h), h)

        return h

    def observe(self, A):
        return jnp.array([self.density(A), self.doubleoccupancy(A)])

    def all(self, A):
        """
        Returns:
            Action and gradient
        """
        act, dact = jax.value_and_grad(self.action, holomorphic=True)(A+0j)
        return act, dact