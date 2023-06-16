from dataclasses import dataclass
from typing import Tuple
from itertools import product
from jax.scipy.linalg import expm

import scipy.special as special
from scipy.optimize import fsolve

import jax
import jax.numpy as jnp
import numpy as np


@dataclass  # 2D model, LxL lattice
class Lattice:
    #    geom: Tuple[int]
    L: int
    nt: int

    def __post_init__(self):
        #       self.D = len(self.geom)
        self.V = self.L**2
        self.dof = self.V * self.nt

    def idx(self, t, x1, x2):
        return (x2 % self.L) + self.L * (x1 % self.L) + self.L * self.L * (t % self.nt)

    def idx1(self, x1, x2):
        return (x2 % self.L) + self.L * (x1 % self.L)

    def idxreverse(self, num):
        t = num // (self.L**2)
        x1 = (num - t * (self.L ** 2)) // self.L
        x2 = (num - t * (self.L ** 2) - x1 * self.L) % self.L

        return jnp.array([t, x1, x2], int)

    def sites(self):
        # Return a list of all sites.
        return jnp.indices((self.nt, self.L, self.L))

    def sites1(self):
        # for exp discretiizaiton
        return jnp.indices((self.nt, self.L, self.L, self.L, self.L))

    def spatial_sites(self):
        # Return a list of spatial sites
        return jnp.indices((self.L, self.L))

    def coords(self, i):
        t = i//self.L
        x = i % self.L
        return t, x

    def even(self):
        e_even = jnp.zeros([self.nt, self.L//2, self.L//2])
        e_odd = jnp.zeros([self.nt, self.L//2, self.L//2])

        def even_update_at(K, t, i, j):
            K = K.at[t, i, j].set(t * self.V + 2*(self.L * i + j))
            return K

        def odd_update_at(K, t, i, j):
            K = K.at[t, i, j].set(
                t * self.V + 2*(self.L * i + j + self.L//2) + 1)
            return K

        ts, xs, ys = jnp.indices((self.nt, self.L//2, self.L//2))
        ts = jnp.ravel(ts)
        xs = jnp.ravel(xs)
        ys = jnp.ravel(ys)

        def even_update_at_i(i, K):
            return even_update_at(K, ts[i], xs[i], ys[i])

        def odd_update_at_i(i, K):
            return odd_update_at(K, ts[i], xs[i], ys[i])

        e_even = jax.lax.fori_loop(0, len(ts), even_update_at_i, e_even)
        e_odd = jax.lax.fori_loop(0, len(ts), odd_update_at_i, e_odd)

        e = jnp.concatenate((e_even, e_odd), axis=None)

        return jnp.array(e, int).sort()

    def odd(self):
        o_even = jnp.zeros([self.nt, self.L//2, self.L//2])
        o_odd = jnp.zeros([self.nt, self.L//2, self.L//2])

        def even_update_at(K, t, i, j):
            K = K.at[t, i, j].set(t * self.V + 2*(self.L * i + j) + 1)
            return K

        def odd_update_at(K, t, i, j):
            K = K.at[t, i, j].set(t * self.V + 2*(self.L * i + j + self.L//2))
            return K

        ts, xs, ys = jnp.indices((self.nt, self.L//2, self.L//2))
        ts = jnp.ravel(ts)
        xs = jnp.ravel(xs)
        ys = jnp.ravel(ys)

        def even_update_at_i(i, K):
            return even_update_at(K, ts[i], xs[i], ys[i])

        def odd_update_at_i(i, K):
            return odd_update_at(K, ts[i], xs[i], ys[i])

        o_even = jax.lax.fori_loop(0, len(ts), even_update_at_i, o_even)
        o_odd = jax.lax.fori_loop(0, len(ts), odd_update_at_i, o_odd)

        o = jnp.concatenate((o_even, o_odd), axis=None)

        return jnp.array(o, int).sort()

    def nearestneighbor(self, num):
        index = self.idxreverse(num)

        x1 = (index+jnp.array([0, 0, 1])
              ) % jnp.array([self.nt, self.L, self.L])
        x2 = (index+jnp.array([0, 1, 0])
              ) % jnp.array([self.nt, self.L, self.L])
        x3 = (index+jnp.array([0, 0, -1])
              ) % jnp.array([self.nt, self.L, self.L])
        x4 = (index+jnp.array([0, -1, 0])
              ) % jnp.array([self.nt, self.L, self.L])

        def _idx(arr):
            return (arr[2] % self.L) + self.L * (arr[1] % self.L) + self.L * self.L * (arr[0] % self.nt)

        return jnp.array([_idx(x1), _idx(x2), _idx(x3), _idx(x4)], int)


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
            if (x1 == x2 and (y1 == (y2 + 1) % size or y1 == (y2 - 1 + size) % size)):
                hop = hop.at[a.idx1(x1, y1), a.idx1(x2, y2)].add(1.0)
            if (y1 == y2 and (x1 == (x2 + 1) % size or x1 == (x2 - 1+size) % size)):
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
    L: int
    nt: int
    Kappa: float
    U: float
    Mu: float
    dt: float

    def __post_init__(self):
        self.lattice = Lattice(self.L, self.nt)
        self.kappa = self.Kappa * self.dt
        self.u = self.U * self.dt
        self.mu = self.Mu * self.dt
        self.beta = self.BetaFunction()

        self.Hopping = Hopping(self.lattice, self.kappa, self.mu)
        self.hopping = self.Hopping.hopping()
        self.h1 = self.Hopping.exp_h1()
        self.h2 = self.Hopping.exp_h2()
        self.dof = self.lattice.dof

        self.periodic_contour = True

    def BetaFunction(self):
        def fn(x):
            return np.real(special.iv(0, np.sqrt((x + 0j)**2 - 1)) /
                           special.iv(0, x + 0j)) - np.exp(-self.u/2)
        betas = fsolve(fn, 1.0)

        return float(betas[0])

    def Hubbard1_old(self, A):
        fer_mat1 = jnp.eye(self.lattice.V) + 0j

        for i in range(self.lattice.nt):
            for x in range(self.lattice.V):
                for y in range(self.lattice.V):
                    fer_mat1 = fer_mat1.at[x, y].multiply(jnp.exp(1j *
                                                                  jnp.sin(A[i * self.lattice.V + x])))

            fer_mat1 = self.h1 @ fer_mat1

        return jnp.eye(self.lattice.V) + fer_mat1

    def Hubbard2_old(self, A):
        fer_mat2 = jnp.eye(self.lattice.V) + 0j

        for i in range(self.lattice.nt):
            for x in range(self.lattice.V):
                for y in range(self.lattice.V):
                    fer_mat2 = fer_mat2.at[x, y].multiply(jnp.exp(-1j *
                                                                  jnp.sin(A[i * self.lattice.V + x])))

            fer_mat2 = self.h2 @ fer_mat2

        return jnp.eye(self.lattice.V) + fer_mat2

    def Hubbard1_new(self, A):
        fer_mat1 = jnp.eye(self.lattice.V) + 0j

        def update_at_tx(t, x, H):
            H = H.at[x, x].add(
                jnp.exp(1j * jnp.sin(A[t * self.lattice.V + x])))
            return H

        def update_at_t(t):
            some = jax.tree_util.Partial(update_at_tx, t)
            temp_mat = jnp.zeros((self.lattice.V, self.lattice.V)) + 0j
            temp_mat = jax.lax.fori_loop(0, self.lattice.V, some, temp_mat)
            return self.h1 @ temp_mat

        def multi(t, fer_mat):
            return update_at_t(t) @ fer_mat

        fer_mat1 = jax.lax.fori_loop(0, self.lattice.nt, multi, fer_mat1)

        return jnp.eye(self.lattice.V) + fer_mat1

    def Hubbard2_new(self, A):
        fer_mat2 = jnp.eye(self.lattice.V) + 0j

        def update_at_tx(t, x, H):
            H = H.at[x, x].add(
                jnp.exp(-1j * jnp.sin(A[t * self.lattice.V + x])))
            return H

        def update_at_t(t):
            some = jax.tree_util.Partial(update_at_tx, t)
            temp_mat = jnp.zeros((self.lattice.V, self.lattice.V)) + 0j
            temp_mat = jax.lax.fori_loop(0, self.lattice.V, some, temp_mat)
            return self.h2 @ temp_mat

        def multi(t, fer_mat):
            return update_at_t(t) @ fer_mat

        fer_mat2 = jax.lax.fori_loop(0, self.lattice.nt, multi, fer_mat2)

        return jnp.eye(self.lattice.V) + fer_mat2

    def Hubbard1_gpu(self, A):  # set if in Hubbard1?
        idx = self.lattice.idx
        idx1 = self.lattice.idx1
        K = jnp.eye(self.lattice.V * self.lattice.nt) + 0j

        def update_at(K, t, x1, x2, y1, y2):
            K = K.at[idx(t, x1, x2), idx(t-1, y1, y2)
                     ].set(self.h1[idx1(x1, x2), idx1(y1, y2)] * jnp.exp(1j * jnp.sin(A[idx(t-1, y1, y2)] + self.mu)))
            return K

        ts, x1s, x2s, y1s, y2s = self.lattice.sites1()
        ts = jnp.ravel(ts)
        x1s = jnp.ravel(x1s)
        x2s = jnp.ravel(x2s)
        y1s = jnp.ravel(y1s)
        y2s = jnp.ravel(y2s)

        def update_at_i(i, K):
            return update_at(K, ts[i], x1s[i], x2s[i], y1s[i], y2s[i])
        K = jax.lax.fori_loop(0, len(ts), update_at_i, K)

        return K

    def Hubbard1(self, A):
        return self.Hubbard1_new(A)

    def Hubbard2(self, A):
        return self.Hubbard2_new(A)

    def action(self, A):
        s1, logdet1 = jnp.linalg.slogdet(self.Hubbard1(A))
        s2, logdet2 = jnp.linalg.slogdet(self.Hubbard2(A))
        return -self.beta * jnp.sum(jnp.cos(A)) - jnp.log(s1) - logdet1 - jnp.log(s2) - logdet2

    def density(self, A):
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
        x = jnp.array(range(size))

        def update_at(x1, y1, x2, y2):
            a = idx(x1, y1)
            b = idx(x2, y2)

            m = - (-1.0)**(x1 + y1 + x2 + y2) * fer_mat1_inv[a, b] * \
                fer_mat1_inv[b, a] + fer_mat2_inv[a, b] * fer_mat2_inv[b, a]

            return m

        def update_at_diagonal(x1, y1):
            a = idx(x1, y1)
            return fer_mat1_inv[a, a] * fer_mat2_inv[a, a]

        m = jax.lax.fori_loop(0, size, lambda i, vx: vx + jax.lax.fori_loop(0, size,
                                                                            lambda j, vy: vy + jax.lax.fori_loop(0, size,
                                                                                                                 lambda k, vz: vz + jax.lax.fori_loop(0, size, lambda l, vw: vw + update_at(x[i], x[j], x[k], x[l]), m), m), m), m)
        m = jax.lax.fori_loop(0, size, lambda i, vx: vx + jax.lax.fori_loop(
            0, size, lambda j, vy: vy + update_at_diagonal(x[i], x[j]), m), m)

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
        x = jnp.array(range(size))

        def update_at(x1, y1, x2, y2):
            a = idx(x1, y1)
            b = idx(x2, y2)

            m = - fer_mat1_inv[a, b] * fer_mat1_inv[b, a] + \
                fer_mat2_inv[a, b] * fer_mat2_inv[b, a]

            return m

        def update_at_diagonal(x1, y1):
            a = idx(x1, y1)
            return fer_mat1_inv[a, a] * fer_mat2_inv[a, a]

        m = jax.lax.fori_loop(0, size, lambda i, vx: vx + jax.lax.fori_loop(0, size,
                                                                            lambda j, vy: vy + jax.lax.fori_loop(0, size,
                                                                                                                 lambda k, vz: vz + jax.lax.fori_loop(0, size, lambda l, vw: vw + update_at(x[i], x[j], x[k], x[l]), m), m), m), m)

        m = jax.lax.fori_loop(0, size, lambda i, vx: vx + jax.lax.fori_loop(
            0, size, lambda j, vy: vy + update_at_diagonal(x[i], x[j]), m), m)

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
        x = jnp.array(range(size))

        def update_at(x1, y1, x2, y2):
            a = idx(x1, y1)
            b = idx(x2, y2)

            h = 0.5 * self.kappa * \
                self.hopping[a, b] * (fer_mat1_inv[a, b] + fer_mat2_inv[a, b])

            return h

        def update_at_diagonal(x1, y1):
            a = idx(x1, y1)
            return fer_mat1_inv[a, a] * fer_mat2_inv[a, a]

        h = jax.lax.fori_loop(0, size, lambda i, vx: vx + jax.lax.fori_loop(0, size,
                                                                            lambda j, vy: vy + jax.lax.fori_loop(0, size,
                                                                                                                 lambda k, vz: vz + jax.lax.fori_loop(0, size, lambda l, vw: vw + update_at(x[i], x[j], x[k], x[l]), h), h), h), h)

        h = jax.lax.fori_loop(0, size, lambda i, vx: vx + jax.lax.fori_loop(
            0, size, lambda j, vy: vy + update_at_diagonal(x[i], x[j]), h), h)

        return h

    def observe(self, A):
        return jnp.array([self.density(A), self.doubleoccupancy(A), self.action(A), self.staggered_magnetization(A)])

    def all(self, A):
        """
        Returns:
            Action and gradient
        """
        act, dact = jax.value_and_grad(self.action, holomorphic=True)(A+0j)
        return act, dact


@dataclass
class ConventionalModel(ImprovedModel):
    L: int
    nt: int
    Kappa: float
    U: float
    Mu: float
    dt: float

    def __post_init__(self):
        self.lattice = Lattice(self.L, self.nt)
        self.kappa = self.Kappa * self.dt
        self.u = self.U * self.dt
        self.mu = self.Mu * self.dt
        self.beta = self.BetaFunction()

        self.Hopping = Hopping(self.lattice, self.kappa, self.mu)
        self.hopping = self.Hopping.hopping()
        self.dof = self.lattice.dof

        self.periodic_contour = True

    def BetaFunction(self):
        def fn(x):
            return np.real(special.iv(1, x + 0j) / (x *
                                                    special.iv(0, x + 0j))) - self.u
        betas = fsolve(fn, 1.0)

        return float(betas[0])

    def Hubbard1_old(self, A):
        idx = self.lattice.idx1
        A = A.reshape((self.lattice.nt, self.lattice.L, self.lattice.L))
        fer_mat1 = jnp.eye(self.lattice.V) + 0j
        H = jnp.zeros((self.lattice.V, self.lattice.V)) + 0j

        for t in range(self.lattice.nt):
            for x1 in range(self.lattice.L):
                for x2 in range(self.lattice.L):
                    H = H.at[idx(x1, x2), idx(x1, x2)].set(-1.0 + self.u /
                                                           2.0 - self.mu - 1j * jnp.sin(A[t, x1, x2]))
                    H = H.at[idx(x1, x2), idx(x1 + 1, x2)].set(-self.kappa)
                    H = H.at[idx(x1, x2), idx(x1, x2 + 1)].set(-self.kappa)
                    H = H.at[idx(x1, x2), idx(x1 - 1, x2)].set(-self.kappa)
                    H = H.at[idx(x1, x2), idx(x1, x2 - 1)].set(-self.kappa)

            fer_mat1 = H @ fer_mat1

        return jnp.eye(self.lattice.V) + fer_mat1

    def Hubbard2_old(self, A):
        idx = self.lattice.idx1
        A = A.reshape((self.lattice.nt, self.lattice.L, self.lattice.L))
        fer_mat2 = jnp.eye(self.lattice.V) + 0j
        H = jnp.zeros((self.lattice.V, self.lattice.V)) + 0j

        for t in range(self.lattice.nt):
            for x1 in range(self.lattice.L):
                for x2 in range(self.lattice.L):
                    H = H.at[idx(x1, x2), idx(x1, x2)].set(-1.0 + self.u /
                                                           2.0 + self.mu + 1j * jnp.sin(A[t, x1, x2]))
                    H = H.at[idx(x1, x2), idx(x1 + 1, x2)].set(-self.kappa)
                    H = H.at[idx(x1, x2), idx(x1, x2 + 1)].set(-self.kappa)
                    H = H.at[idx(x1, x2), idx(x1 - 1, x2)].set(-self.kappa)
                    H = H.at[idx(x1, x2), idx(x1, x2 - 1)].set(-self.kappa)

            fer_mat2 = H @ fer_mat2

        return jnp.eye(self.lattice.V) + fer_mat2

    def Hubbard1_new(self, A):
        idx = self.lattice.idx1
        A = A.reshape((self.lattice.nt, self.lattice.L, self.lattice.L))
        fer_mat1 = jnp.eye(self.lattice.V) + 0j

        def update_at_tx(t, x1, x2, H):
            H = H.at[idx(x1, x2), idx(x1, x2)].add(-1.0 + self.u /
                                                   2.0 - self.mu - 1j * jnp.sin(A[t, x1, x2]))
            H = H.at[idx(x1, x2), idx(x1 + 1, x2)].add(-self.kappa)
            H = H.at[idx(x1, x2), idx(x1, x2 + 1)].add(-self.kappa)
            H = H.at[idx(x1, x2), idx(x1 - 1, x2)].add(-self.kappa)
            H = H.at[idx(x1, x2), idx(x1, x2 - 1)].add(-self.kappa)

            return H

        x1s, x2s = self.lattice.spatial_sites()
        x1s = jnp.ravel(x1s)
        x2s = jnp.ravel(x2s)

        def update_at_ti(t, i, H):
            return update_at_tx(t, x1s[i], x2s[i], H)

        def update_at_t(t):
            temp_mat = jnp.zeros((self.lattice.V, self.lattice.V)) + 0j
            func = jax.tree_util.Partial(update_at_ti, t)

            temp_mat = jax.lax.fori_loop(0, len(x1s), func, temp_mat)
            return temp_mat

        def multi(t, fer_mat):
            return update_at_t(t) @ fer_mat

        fer_mat1 = jax.lax.fori_loop(0, self.lattice.nt, multi, fer_mat1)

        return jnp.eye(self.lattice.V) + fer_mat1

    def Hubbard2_new(self, A):
        idx = self.lattice.idx1
        A = A.reshape((self.lattice.nt, self.lattice.L, self.lattice.L))
        fer_mat2 = jnp.eye(self.lattice.V) + 0j

        def update_at_tx(t, x1, x2, H):
            H = H.at[idx(x1, x2), idx(x1, x2)].add(-1.0 + self.u /
                                                   2.0 + self.mu + 1j * jnp.sin(A[t, x1, x2]))
            H = H.at[idx(x1, x2), idx(x1 + 1, x2)].add(-self.kappa)
            H = H.at[idx(x1, x2), idx(x1, x2 + 1)].add(-self.kappa)
            H = H.at[idx(x1, x2), idx(x1 - 1, x2)].add(-self.kappa)
            H = H.at[idx(x1, x2), idx(x1, x2 - 1)].add(-self.kappa)

            return H

        x1s, x2s = self.lattice.spatial_sites()
        x1s = jnp.ravel(x1s)
        x2s = jnp.ravel(x2s)

        def update_at_ti(t, i, H):
            return update_at_tx(t, x1s[i], x2s[i], H)

        def update_at_t(t):
            temp_mat = jnp.zeros((self.lattice.V, self.lattice.V)) + 0j
            func = jax.tree_util.Partial(update_at_ti, t)

            temp_mat = jax.lax.fori_loop(0, len(x1s), func, temp_mat)
            return temp_mat

        def multi(t, fer_mat):
            return update_at_t(t) @ fer_mat

        fer_mat2 = jax.lax.fori_loop(0, self.lattice.nt, multi, fer_mat2)

        return jnp.eye(self.lattice.V) + fer_mat2

    def Hubbard1(self, A):
        return self.Hubbard1_new(A)

    def Hubbard2(self, A):
        return self.Hubbard2_new(A)

    def action(self, A):
        s1, logdet1 = jnp.linalg.slogdet(self.Hubbard1(A))
        s2, logdet2 = jnp.linalg.slogdet(self.Hubbard2(A))
        return -self.beta * jnp.sum(jnp.cos(A)) - jnp.log(s1) - logdet1 - jnp.log(s2) - logdet2


@dataclass
class ImprovedGaussianModel(ImprovedModel):
    L: int
    nt: int
    Kappa: float
    U: float
    Mu: float
    dt: float

    def __post_init__(self):
        self.lattice = Lattice(self.L, self.nt)
        self.kappa = self.Kappa * self.dt
        self.u = self.U * self.dt
        self.mu = self.Mu * self.dt

        self.Hopping = Hopping(self.lattice, self.kappa, self.mu)
        self.hopping = self.Hopping.hopping()
        self.h1 = self.Hopping.exp_h1()
        self.h2 = self.Hopping.exp_h2()
        self.dof = self.lattice.dof

        self.periodic_contour = False

    def Hubbard1(self, A):
        fer_mat1 = jnp.eye(self.lattice.V) + 0j

        def update_at_tx(t, x, H):
            H = H.at[x, x].add(
                jnp.exp(1j * A[t * self.lattice.V + x]))
            return H

        def update_at_t(t):
            some = jax.tree_util.Partial(update_at_tx, t)
            temp_mat = jnp.zeros((self.lattice.V, self.lattice.V)) + 0j
            temp_mat = jax.lax.fori_loop(0, self.lattice.V, some, temp_mat)
            return self.h1 @ temp_mat

        def multi(t, fer_mat):
            return update_at_t(t) @ fer_mat

        fer_mat1 = jax.lax.fori_loop(0, self.lattice.nt, multi, fer_mat1)

        return jnp.eye(self.lattice.V) + fer_mat1

    def Hubbard2(self, A):
        fer_mat2 = jnp.eye(self.lattice.V) + 0j

        def update_at_tx(t, x, H):
            H = H.at[x, x].add(
                jnp.exp(-1j * A[t * self.lattice.V + x]))
            return H

        def update_at_t(t):
            some = jax.tree_util.Partial(update_at_tx, t)
            temp_mat = jnp.zeros((self.lattice.V, self.lattice.V)) + 0j
            temp_mat = jax.lax.fori_loop(0, self.lattice.V, some, temp_mat)
            return self.h2 @ temp_mat

        def multi(t, fer_mat):
            return update_at_t(t) @ fer_mat

        fer_mat2 = jax.lax.fori_loop(0, self.lattice.nt, multi, fer_mat2)

        return jnp.eye(self.lattice.V) + fer_mat2

    def action(self, A):
        s1, logdet1 = jnp.linalg.slogdet(self.Hubbard1(A))
        s2, logdet2 = jnp.linalg.slogdet(self.Hubbard2(A))
        return jnp.sum(A ** 2) / (2 * self.u) - jnp.log(s1) - logdet1 - jnp.log(s2) - logdet2


@dataclass
class ConventionalGaussianModel(ImprovedGaussianModel):
    L: int
    nt: int
    Kappa: float
    U: float
    Mu: float
    dt: float

    def __post_init__(self):
        self.lattice = Lattice(self.L, self.nt)
        self.kappa = self.Kappa * self.dt
        self.u = self.U * self.dt
        self.mu = self.Mu * self.dt

        self.Hopping = Hopping(self.lattice, self.kappa, self.mu)
        self.hopping = self.Hopping.hopping()
        self.dof = self.lattice.dof

        self.periodic_contour = False

    def Hubbard1(self, A):
        idx = self.lattice.idx1
        A = A.reshape((self.lattice.nt, self.lattice.L, self.lattice.L))
        fer_mat1 = jnp.eye(self.lattice.V) + 0j

        def update_at_tx(t, x1, x2, H):
            H = H.at[idx(x1, x2), idx(x1, x2)].add(-1.0 + self.u /
                                                   2.0 - self.mu - 1j * A[t, x1, x2])
            H = H.at[idx(x1, x2), idx(x1 + 1, x2)].add(-self.kappa)
            H = H.at[idx(x1, x2), idx(x1, x2 + 1)].add(-self.kappa)
            H = H.at[idx(x1, x2), idx(x1 - 1, x2)].add(-self.kappa)
            H = H.at[idx(x1, x2), idx(x1, x2 - 1)].add(-self.kappa)

            return H

        x1s, x2s = self.lattice.spatial_sites()
        x1s = jnp.ravel(x1s)
        x2s = jnp.ravel(x2s)

        def update_at_ti(t, i, H):
            return update_at_tx(t, x1s[i], x2s[i], H)

        def update_at_t(t):
            temp_mat = jnp.zeros((self.lattice.V, self.lattice.V)) + 0j
            func = jax.tree_util.Partial(update_at_ti, t)

            temp_mat = jax.lax.fori_loop(0, len(x1s), func, temp_mat)
            return temp_mat

        def multi(t, fer_mat):
            return update_at_t(t) @ fer_mat

        fer_mat1 = jax.lax.fori_loop(0, self.lattice.nt, multi, fer_mat1)

        return jnp.eye(self.lattice.V) + fer_mat1

    def Hubbard2(self, A):
        idx = self.lattice.idx1
        A = A.reshape((self.lattice.nt, self.lattice.L, self.lattice.L))
        fer_mat2 = jnp.eye(self.lattice.V) + 0j

        def update_at_tx(t, x1, x2, H):
            H = H.at[idx(x1, x2), idx(x1, x2)].add(-1.0 + self.u /
                                                   2.0 + self.mu + 1j * A[t, x1, x2])
            H = H.at[idx(x1, x2), idx(x1 + 1, x2)].add(-self.kappa)
            H = H.at[idx(x1, x2), idx(x1, x2 + 1)].add(-self.kappa)
            H = H.at[idx(x1, x2), idx(x1 - 1, x2)].add(-self.kappa)
            H = H.at[idx(x1, x2), idx(x1, x2 - 1)].add(-self.kappa)

            return H

        x1s, x2s = self.lattice.spatial_sites()
        x1s = jnp.ravel(x1s)
        x2s = jnp.ravel(x2s)

        def update_at_ti(t, i, H):
            return update_at_tx(t, x1s[i], x2s[i], H)

        def update_at_t(t):
            temp_mat = jnp.zeros((self.lattice.V, self.lattice.V)) + 0j
            func = jax.tree_util.Partial(update_at_ti, t)

            temp_mat = jax.lax.fori_loop(0, len(x1s), func, temp_mat)
            return temp_mat

        def multi(t, fer_mat):
            return update_at_t(t) @ fer_mat

        fer_mat2 = jax.lax.fori_loop(0, self.lattice.nt, multi, fer_mat2)

        return jnp.eye(self.lattice.V) + fer_mat2

    def action(self, A):
        s1, logdet1 = jnp.linalg.slogdet(self.Hubbard1(A))
        s2, logdet2 = jnp.linalg.slogdet(self.Hubbard2(A))
        return jnp.sum(A ** 2) / (2 * self.u) - jnp.log(s1) - logdet1 - jnp.log(s2) - logdet2


@dataclass
class DiagonalModel(ImprovedModel):
    L: int
    nt: int
    Kappa: float
    U: float
    Mu: float
    dt: float

    def __post_init__(self):
        self.lattice = Lattice(self.L, self.nt)
        self.kappa = self.Kappa * self.dt
        self.u = self.U * self.dt
        self.mu = self.Mu * self.dt

        self.Hopping = Hopping(self.lattice, self.kappa, self.mu)
        self.hopping = self.Hopping.hopping()
        self.dof = self.lattice.dof

        self.periodic_contour = False

        self.hopping_mat = jnp.eye(self.lattice.V) - self.kappa * self.hopping
        self.hopping_inverse = jnp.linalg.inv(self.hopping_mat)
        self.hopping_s, self.hopping_logdet = jnp.linalg.slogdet(
            self.hopping_mat)

    def Hubbard1(self, A):
        fer_mat1 = self.hopping_mat + 0j

        def update_at_tx(t, x, H):
            H = H.at[x, x].add(
                -jnp.exp(1j * A[t * self.lattice.V + x]) - self.mu)
            return H

        def update_at_t(t):
            some = jax.tree_util.Partial(update_at_tx, t)
            temp_mat = jnp.zeros((self.lattice.V, self.lattice.V)) + 0j
            temp_mat = jax.lax.fori_loop(0, self.lattice.V, some, temp_mat)
            return temp_mat @ self.hopping_inverse

        def multi(t, fer_mat):
            return update_at_t(t) @ fer_mat

        fer_mat1 = jax.lax.fori_loop(0, self.lattice.nt, multi, fer_mat1)

        return jnp.eye(self.lattice.V) + fer_mat1

    def Hubbard2(self, A):
        fer_mat2 = self.hopping_mat + 0j

        def update_at_tx(t, x, H):
            H = H.at[x, x].add(
                -jnp.exp(-1j * A[t * self.lattice.V + x]) + self.mu)
            return H

        def update_at_t(t):
            some = jax.tree_util.Partial(update_at_tx, t)
            temp_mat = jnp.zeros((self.lattice.V, self.lattice.V)) + 0j
            temp_mat = jax.lax.fori_loop(0, self.lattice.V, some, temp_mat)
            return temp_mat @ self.hopping_inverse

        def multi(t, fer_mat):
            return update_at_t(t) @ fer_mat

        fer_mat2 = jax.lax.fori_loop(0, self.lattice.nt, multi, fer_mat2)

        return jnp.eye(self.lattice.V) + fer_mat2

    def action(self, A):
        s1, logdet1 = jnp.linalg.slogdet(self.Hubbard1(A))
        s2, logdet2 = jnp.linalg.slogdet(self.Hubbard2(A))
        return jnp.sum(A ** 2) / (2 * self.u) - 2 * self.lattice.nt * (self.hopping_s + self.hopping_logdet) - jnp.log(s1) - logdet1 - jnp.log(s2) - logdet2


@dataclass
class ImprovedGaussianAlphaModel(ImprovedGaussianModel):
    L: int
    nt: int
    Kappa: float
    U: float
    Mu: float
    dt: float
    alpha: float

    def __post_init__(self):
        self.lattice = Lattice(self.L, self.nt)
        self.lattice.dof = 2 * self.lattice.dof
        self.kappa = self.Kappa * self.dt
        self.u = self.U * self.dt
        self.mu = self.Mu * self.dt

        self.Hopping = Hopping(self.lattice, self.kappa, self.mu)
        self.hopping = self.Hopping.hopping()
        self.h1 = self.Hopping.exp_h1()
        self.h2 = self.Hopping.exp_h2()
        self.dof = self.lattice.dof

        self.periodic_contour = False

    def Hubbard1(self, A):
        fer_mat1 = jnp.eye(self.lattice.V) + 0j
        phi = A[:self.lattice.dof//2]
        chi = A[self.lattice.dof//2:]

        def update_at_tx(t, x, H):
            H = H.at[x, x].add(
                jnp.exp(1j * phi[t * self.lattice.V + x] + chi[t * self.lattice.V + x]))
            return H

        def update_at_t(t):
            some = jax.tree_util.Partial(update_at_tx, t)
            temp_mat = jnp.zeros((self.lattice.V, self.lattice.V)) + 0j
            temp_mat = jax.lax.fori_loop(0, self.lattice.V, some, temp_mat)
            return self.h1 @ temp_mat

        def multi(t, fer_mat):
            return update_at_t(t) @ fer_mat

        fer_mat1 = jax.lax.fori_loop(0, self.lattice.nt, multi, fer_mat1)

        return jnp.eye(self.lattice.V) + fer_mat1

    def Hubbard2(self, A):
        fer_mat2 = jnp.eye(self.lattice.V) + 0j
        phi = A[:self.lattice.dof//2]
        chi = A[self.lattice.dof//2:]

        def update_at_tx(t, x, H):
            H = H.at[x, x].add(
                jnp.exp(-1j * phi[t * self.lattice.V + x] + chi[t * self.lattice.V + x]))
            return H

        def update_at_t(t):
            some = jax.tree_util.Partial(update_at_tx, t)
            temp_mat = jnp.zeros((self.lattice.V, self.lattice.V)) + 0j
            temp_mat = jax.lax.fori_loop(0, self.lattice.V, some, temp_mat)
            return self.h2 @ temp_mat

        def multi(t, fer_mat):
            return update_at_t(t) @ fer_mat

        fer_mat2 = jax.lax.fori_loop(0, self.lattice.nt, multi, fer_mat2)

        return jnp.eye(self.lattice.V) + fer_mat2

    def action(self, A):
        s1, logdet1 = jnp.linalg.slogdet(self.Hubbard1(A))
        s2, logdet2 = jnp.linalg.slogdet(self.Hubbard2(A))
        return jnp.sum(A[:self.lattice.dof//2] ** 2) / (2 * self.alpha * self.u) + jnp.sum((A[self.lattice.dof//2:] - (1 - self.alpha) * self.u)**2)/(2 * (1 - self.alpha) * self.u) - jnp.log(s1) - logdet1 - jnp.log(s2) - logdet2


@dataclass
class HyperbolicModel(ImprovedGaussianModel):
    L: int
    nt: int
    Kappa: float
    U: float
    Mu: float
    dt: float

    def __post_init__(self):
        self.lattice = Lattice(self.L, self.nt)
        self.kappa = self.Kappa * self.dt
        self.u = self.U * self.dt
        self.mu = self.Mu * self.dt
        self.beta = self.BetaFunction()

        self.Hopping = Hopping(self.lattice, self.kappa, self.mu)
        self.hopping = self.Hopping.hopping()
        self.h1 = self.Hopping.exp_h1()
        self.h2 = self.Hopping.exp_h2()
        self.dof = self.lattice.dof

        self.periodic_contour = False

    def BetaFunction(self):
        def fn(x):
            return np.real(special.spherical_jn(0, x + 0j)) - np.exp(-self.u/2)
        betas = fsolve(fn, 1.0)

        return float(betas[0])

    def Hubbard1(self, A):
        fer_mat1 = jnp.eye(self.lattice.V) + 0j

        def update_at_tx(t, x, H):
            H = H.at[x, x].add(
                jnp.exp(1j * self.beta * A[t * self.lattice.V + x]))
            return H

        def update_at_t(t):
            some = jax.tree_util.Partial(update_at_tx, t)
            temp_mat = jnp.zeros((self.lattice.V, self.lattice.V)) + 0j
            temp_mat = jax.lax.fori_loop(0, self.lattice.V, some, temp_mat)
            return self.h1 @ temp_mat

        def multi(t, fer_mat):
            return update_at_t(t) @ fer_mat

        fer_mat1 = jax.lax.fori_loop(0, self.lattice.nt, multi, fer_mat1)

        return jnp.eye(self.lattice.V) + fer_mat1

    def Hubbard2(self, A):
        fer_mat2 = jnp.eye(self.lattice.V) + 0j

        def update_at_tx(t, x, H):
            H = H.at[x, x].add(
                jnp.exp(-1j * self.beta * A[t * self.lattice.V + x]))
            return H

        def update_at_t(t):
            some = jax.tree_util.Partial(update_at_tx, t)
            temp_mat = jnp.zeros((self.lattice.V, self.lattice.V)) + 0j
            temp_mat = jax.lax.fori_loop(0, self.lattice.V, some, temp_mat)
            return self.h2 @ temp_mat

        def multi(t, fer_mat):
            return update_at_t(t) @ fer_mat

        fer_mat2 = jax.lax.fori_loop(0, self.lattice.nt, multi, fer_mat2)

        return jnp.eye(self.lattice.V) + fer_mat2

    def action(self, A):
        s1, logdet1 = jnp.linalg.slogdet(self.Hubbard1(jnp.tanh(A)))
        s2, logdet2 = jnp.linalg.slogdet(self.Hubbard2(jnp.tanh(A)))
        return 2 * jnp.sum(jnp.log(jnp.cosh(A))) - jnp.log(s1) - logdet1 - jnp.log(s2) - logdet2


@dataclass
class ImprovedGaussianSpinModel(ImprovedModel):
    L: int
    nt: int
    Kappa: float
    U: float
    Mu: float
    dt: float

    def __post_init__(self):
        self.lattice = Lattice(self.L, self.nt)
        self.kappa = self.Kappa * self.dt
        self.u = self.U * self.dt
        self.mu = self.Mu * self.dt

        self.Hopping = Hopping(self.lattice, self.kappa, self.mu)
        self.hopping = self.Hopping.hopping()

        self.h1 = self.kappa * self.hopping
        for i in range(self.lattice.L**2):
            self.h1 = self.h1.at[i, i].add(self.mu-self.u)
        self.h1 = expm(self.h1)

        self.h2 = self.kappa * self.hopping
        for i in range(self.lattice.L**2):
            self.h2 = self.h2.at[i, i].add(-self.mu-self.u)
        self.h2 = expm(self.h2)

        self.dof = self.lattice.dof

        self.periodic_contour = False

    def Hubbard1(self, A):
        fer_mat1 = jnp.eye(self.lattice.V) + 0j

        def update_at_tx(t, x, H):
            H = H.at[x, x].add(
                jnp.exp(A[t * self.lattice.V + x]))
            return H

        def update_at_t(t):
            some = jax.tree_util.Partial(update_at_tx, t)
            temp_mat = jnp.zeros((self.lattice.V, self.lattice.V)) + 0j
            temp_mat = jax.lax.fori_loop(0, self.lattice.V, some, temp_mat)
            return self.h1 @ temp_mat

        def multi(t, fer_mat):
            return update_at_t(t) @ fer_mat

        fer_mat1 = jax.lax.fori_loop(0, self.lattice.nt, multi, fer_mat1)

        return jnp.eye(self.lattice.V) + fer_mat1

    def Hubbard2(self, A):
        fer_mat2 = jnp.eye(self.lattice.V) + 0j

        def update_at_tx(t, x, H):
            H = H.at[x, x].add(
                jnp.exp(A[t * self.lattice.V + x]))
            return H

        def update_at_t(t):
            some = jax.tree_util.Partial(update_at_tx, t)
            temp_mat = jnp.zeros((self.lattice.V, self.lattice.V)) + 0j
            temp_mat = jax.lax.fori_loop(0, self.lattice.V, some, temp_mat)
            return self.h2 @ temp_mat

        def multi(t, fer_mat):
            return update_at_t(t) @ fer_mat

        fer_mat2 = jax.lax.fori_loop(0, self.lattice.nt, multi, fer_mat2)

        return jnp.eye(self.lattice.V) + fer_mat2

    def action(self, A):
        s1, logdet1 = jnp.linalg.slogdet(self.Hubbard1(A))
        s2, logdet2 = jnp.linalg.slogdet(self.Hubbard2(A))
        return jnp.sum(A ** 2) / (2 * self.u) - jnp.log(s1) - logdet1 - jnp.log(s2) - logdet2


@dataclass
class ConventionalSpinModel(ImprovedModel):
    L: int
    nt: int
    Kappa: float
    U: float
    Mu: float
    dt: float

    def __post_init__(self):
        self.lattice = Lattice(self.L, self.nt)
        self.kappa = self.Kappa * self.dt
        self.u = self.U * self.dt
        self.mu = self.Mu * self.dt
        self.beta = self.BetaFunction()

        self.Hopping = Hopping(self.lattice, self.kappa, self.mu)
        self.hopping = self.Hopping.hopping()
        self.dof = self.lattice.dof

        self.periodic_contour = True

    def BetaFunction(self):
        def fn(x):
            return np.real(special.iv(1, x + 0j) / (x *
                                                    special.iv(0, x + 0j))) - self.u
        betas = fsolve(fn, 1.0)

        return float(betas[0])

    def Hubbard1(self, A):
        idx = self.lattice.idx1
        A = A.reshape((self.lattice.nt, self.lattice.L, self.lattice.L))
        fer_mat1 = jnp.eye(self.lattice.V) + 0j

        def update_at_tx(t, x1, x2, H):
            H = H.at[idx(x1, x2), idx(x1, x2)].add(-1.0 + self.u /
                                                   2.0 - self.mu + jnp.sin(A[t, x1, x2]))
            H = H.at[idx(x1, x2), idx(x1 + 1, x2)].add(-self.kappa)
            H = H.at[idx(x1, x2), idx(x1, x2 + 1)].add(-self.kappa)
            H = H.at[idx(x1, x2), idx(x1 - 1, x2)].add(-self.kappa)
            H = H.at[idx(x1, x2), idx(x1, x2 - 1)].add(-self.kappa)

            return H

        x1s, x2s = self.lattice.spatial_sites()
        x1s = jnp.ravel(x1s)
        x2s = jnp.ravel(x2s)

        def update_at_ti(t, i, H):
            return update_at_tx(t, x1s[i], x2s[i], H)

        def update_at_t(t):
            temp_mat = jnp.zeros((self.lattice.V, self.lattice.V)) + 0j
            func = jax.tree_util.Partial(update_at_ti, t)

            temp_mat = jax.lax.fori_loop(0, len(x1s), func, temp_mat)
            return temp_mat

        def multi(t, fer_mat):
            return update_at_t(t) @ fer_mat

        fer_mat1 = jax.lax.fori_loop(0, self.lattice.nt, multi, fer_mat1)

        return jnp.eye(self.lattice.V) + fer_mat1

    def Hubbard2(self, A):
        idx = self.lattice.idx1
        A = A.reshape((self.lattice.nt, self.lattice.L, self.lattice.L))
        fer_mat2 = jnp.eye(self.lattice.V) + 0j

        def update_at_tx(t, x1, x2, H):
            H = H.at[idx(x1, x2), idx(x1, x2)].add(-1.0 + self.u /
                                                   2.0 + self.mu + jnp.sin(A[t, x1, x2]))
            H = H.at[idx(x1, x2), idx(x1 + 1, x2)].add(-self.kappa)
            H = H.at[idx(x1, x2), idx(x1, x2 + 1)].add(-self.kappa)
            H = H.at[idx(x1, x2), idx(x1 - 1, x2)].add(-self.kappa)
            H = H.at[idx(x1, x2), idx(x1, x2 - 1)].add(-self.kappa)

            return H

        x1s, x2s = self.lattice.spatial_sites()
        x1s = jnp.ravel(x1s)
        x2s = jnp.ravel(x2s)

        def update_at_ti(t, i, H):
            return update_at_tx(t, x1s[i], x2s[i], H)

        def update_at_t(t):
            temp_mat = jnp.zeros((self.lattice.V, self.lattice.V)) + 0j
            func = jax.tree_util.Partial(update_at_ti, t)

            temp_mat = jax.lax.fori_loop(0, len(x1s), func, temp_mat)
            return temp_mat

        def multi(t, fer_mat):
            return update_at_t(t) @ fer_mat

        fer_mat2 = jax.lax.fori_loop(0, self.lattice.nt, multi, fer_mat2)

        return jnp.eye(self.lattice.V) + fer_mat2

    def action(self, A):
        s1, logdet1 = jnp.linalg.slogdet(self.Hubbard1(A))
        s2, logdet2 = jnp.linalg.slogdet(self.Hubbard2(A))
        return -self.beta * jnp.sum(jnp.cos(A)) - jnp.log(s1) - logdet1 - jnp.log(s2) - logdet2
