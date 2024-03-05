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
        self.shape = (self.nt, self.L, self.L)

    def idx(self, *args):
        n = len(args)
        return jnp.ravel_multi_index(args, self.shape[-n:], mode='wrap')

    def cartidx(self, idx):
        return jnp.unravel_index(idx, self.shape)

    def sites(self):
        # Return a list of all sites.
        return jnp.indices(self.shape)

    def spatial_sites(self):
        # Return a list of spatial sites
        return jnp.indices((self.L, self.L))

    def coords(self, i):
        return jnp.unravel_index(i, (self.nt, self.L))

    def even(self):
        ts, xs, ys = jnp.indices((self.nt, self.L//2, self.L//2))
        e_even = ts * self.V + 2*(self.L * xs + ys)
        e_odd = ts * self.V + 2*(self.L * xs + ys + self.L//2) + 1
        e = jnp.concatenate((e_even, e_odd), axis=None)
        return jnp.array(e, int).sort()

    def odd(self):
        ts, xs, ys = jnp.indices((self.nt, self.L//2, self.L//2))
        o_even = ts * self.V + 2*(self.L * xs + ys) + 1
        o_odd = ts * self.V + 2*(self.L * xs + ys + self.L//2)
        o = jnp.concatenate((o_even, o_odd), axis=None)
        return jnp.array(o, int).sort()

    def nearestneighbor(self, idx):
        t, x, y = self.idxcart(idx)
        indices = ((t, x, y+1), (t, x+1, y), (t, x, y-1), (t, x-1, y))
        return jnp.ravel_multi_index(indices, self.shape, mode='wrap')


@dataclass
class Hopping:
    lattice: Lattice
    kappa: float
    mu: float

    def hopping(self):
        L = self.lattice.L
        x1, y1, x2, y2 = jnp.indices((L, L, L, L))
        pred = (((x1 == x2) & ((y1 == (y2 + 1) % L) | (y1 == (y2 - 1) % L))) |
                ((y1 == y2) & ((x1 == (x2 + 1) % L) | (x1 == (x2 - 1) % L))))
        hop = jnp.where(pred, 1.0, 0.0)
        return hop.reshape((L*L, L*L))

    def hopping2(self):
        L = self.lattice.L
        x1, y1, x2, y2 = jnp.indices((L, L, L, L))
        pred = (((x1 == (x2 + 1) % L) & ((y1 == (y2 + 1) % L) | (y1 == (y2 - 1) % L))) |
                ((x1 == (x2 - 1) % L) & ((y1 == (y2 + 1) % L) | (y1 == (y2 - 1) % L))))
        hop = jnp.where(pred, 1.0, 0.0)
        return hop.reshape((L*L, L*L))

    def exp_h1(self):
        h1 = self.kappa * self.hopping() + self.mu * jnp.identity(self.lattice.V) + \
            0.0 * self.hopping2()
        h1 = expm(h1)
        return h1

    def exp_h2(self):
        h2 = self.kappa * self.hopping()
        h2 = self.kappa * self.hopping() - self.mu * jnp.identity(self.lattice.V) + \
            0.0 * self.hopping2()
        h2 = expm(h2)
        return h2

    def exp_h(self):
        h = self.kappa * self.hopping() + 0.0 * self.hopping2()
        h = expm(h)
        return h


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
        self.h = {1: self.Hopping.exp_h1(), -1: self.Hopping.exp_h2()}

        self.h1_svd = jnp.linalg.svd(self.h[1])
        self.h2_svd = jnp.linalg.svd(self.h[-1])

        self.dof = self.lattice.dof
        self.periodic_contour = True

    def BetaFunction(self):
        def fn(x):
            return np.real(special.iv(0, np.sqrt((x + 0j)**2 - 1)) / special.iv(0, x + 0j)) - np.exp(-self.u/2)
        betas = fsolve(fn, 1.0)

        return float(betas[0])

    def Hubbard(self, A, spin=1):
        M = jnp.identity(self.L*self.L) + 0j
        Ab = A.reshape((self.nt, self.L*self.L))
        for t in range(self.nt):
            i, _ = jnp.diag_indices_from(M)
            M = self.h[spin] @ jnp.diag(jnp.exp(1j*spin*jnp.sin(Ab[t, i]))) @ M
        return jnp.identity(self.L*self.L) + M

    def action(self, A):
        s1, logdet1 = jnp.linalg.slogdet(self.Hubbard(A))
        s2, logdet2 = jnp.linalg.slogdet(self.Hubbard(A, -1))
        return -self.beta * jnp.sum(jnp.cos(A)) - jnp.log(s1) - logdet1 - jnp.log(s2) - logdet2

    # Error: Singular value decomposition JVP not implemented for full matrices
    # https://github.com/google/jax/issues/2011
    def svd_mult(self, A, B):  # fact_mult
        m = jnp.linalg.svd(jnp.diag(A[1])@A[2]@B[0]@jnp.diag(B[1]))
        u = A[0] @ m[0]
        s = m[1]
        v = m[2] @ B[2]

        return u, s, v

    def Hubbard1_svd(self, A):
        fer_mat1 = (jnp.eye(self.lattice.V) + 0j,
                    jnp.ones(self.lattice.V), jnp.eye(self.lattice.V)+0j)

        def update_at_tx(t, x, H):  # don't touch
            H = H.at[x, x].add(
                jnp.exp(1j * jnp.sin(A[t * self.lattice.V + x])))
            return H

        def update_at_t(t):  # don't touch. generate B_t
            some = jax.tree_util.Partial(update_at_tx, t)
            temp_mat = jnp.zeros((self.lattice.V, self.lattice.V)) + 0j
            temp_mat = jax.lax.fori_loop(0, self.lattice.V, some, temp_mat)
            return jnp.linalg.svd(temp_mat)

        def multi(t, fer_mat):
            m1 = self.svd_mult(update_at_t(t), fer_mat)
            return self.svd_mult(self.h1_svd, m1)

        fer_mat1 = jax.lax.fori_loop(0, self.lattice.nt, multi, fer_mat1)

        final_svd = jnp.linalg.svd(
            fer_mat1[0].conj().T @ fer_mat1[2].conj().T + jnp.diag(fer_mat1[1]))
        final_u = fer_mat1[0] @ final_svd[0]
        final_d = final_svd[1]
        final_v = final_svd[2] @ fer_mat1[2]

        return final_u, final_d, final_v

    def Hubbard2_svd(self, A):
        fer_mat2 = (jnp.eye(self.lattice.V) + 0j,
                    jnp.ones(self.lattice.V), jnp.eye(self.lattice.V)+0j)

        def update_at_tx(t, x, H):  # don't touch
            H = H.at[x, x].add(
                jnp.exp(1j * jnp.sin(A[t * self.lattice.V + x])))
            return H

        def update_at_t(t):  # don't touch. generate B_t
            some = jax.tree_util.Partial(update_at_tx, t)
            temp_mat = jnp.zeros((self.lattice.V, self.lattice.V)) + 0j
            temp_mat = jax.lax.fori_loop(0, self.lattice.V, some, temp_mat)
            return jnp.linalg.svd(temp_mat)

        def multi(t, fer_mat):
            m1 = self.svd_mult(update_at_t(t), fer_mat)
            return self.svd_mult(self.h2_svd, m1)

        fer_mat2 = jax.lax.fori_loop(0, self.lattice.nt, multi, fer_mat2)

        final_svd = jnp.linalg.svd(
            fer_mat2[0].conj().T @ fer_mat2[2].conj().T + jnp.diag(fer_mat2[1]))
        final_u = fer_mat2[0] @ final_svd[0]
        final_d = final_svd[1]
        final_v = final_svd[2] @ fer_mat2[2]

        return final_u, final_d, final_v

    def action_svd(self, A):
        u1, d1, v1 = self.Hubbard1_svd(A)
        u1_s, u1_logdet = jnp.linalg.slogdet(u1)
        d1_logdet = jnp.sum(jnp.log(d1))
        v1_s, v1_logdet = jnp.linalg.slogdet(v1)
        logdet1 = u1_s + u1_logdet + d1_logdet + v1_s + v1_logdet

        u2, d2, v2 = self.Hubbard2_svd(A)
        u2_s, u2_logdet = jnp.linalg.slogdet(u2)
        d2_logdet = jnp.sum(jnp.log(d2))
        v2_s, v2_logdet = jnp.linalg.slogdet(v2)
        logdet2 = u2_s + u2_logdet + d2_logdet + v2_s + v2_logdet

        return -self.beta * jnp.sum(jnp.cos(A)) - logdet1 - logdet2

    def density(self, A):
        fer_mat1_inv = jnp.linalg.inv(self.Hubbard(A))
        fer_mat2_inv = jnp.linalg.inv(self.Hubbard(A, -1))

        return jnp.trace(fer_mat2_inv - fer_mat1_inv) / self.lattice.V

    def doubleoccupancy(self, A):
        D1 = jnp.diagonal(jnp.linalg.inv(self.Hubbard(A)))
        D2 = jnp.diagonal(jnp.linalg.inv(self.Hubbard(A, -1)))
        return jnp.mean(D2*(1-D1))

    def staggered_magnetization(self, A):
        m = 0 + 0j
        fer_mat1_inv = jnp.linalg.inv(self.Hubbard(A, 1))
        fer_mat2_inv = jnp.linalg.inv(self.Hubbard(A, -1))

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
        fer_mat1_inv = jnp.linalg.inv(self.Hubbard(A, 1))
        fer_mat2_inv = jnp.linalg.inv(self.Hubbard(A, -1))

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
        fer_mat1_inv = jnp.linalg.inv(self.Hubbard(A))

        return jnp.trace(fer_mat1_inv) / self.lattice.V

    def n2(self, A):
        fer_mat2_inv = jnp.linalg.inv(self.Hubbard(A, -1))

        return jnp.trace(fer_mat2_inv) / self.lattice.V

    def hamiltonian(self, A):
        h = 0 + 0j
        fer_mat1_inv = jnp.linalg.inv(self.Hubbard(A))
        fer_mat2_inv = jnp.linalg.inv(self.Hubbard(A, -1))

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
        return jnp.array([self.density(A), self.doubleoccupancy(A), self.action(A)])


@dataclass
class ImprovedModel2(ImprovedModel):
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
        self.h = self.Hopping.exp_h()

        self.dof = self.lattice.dof
        self.periodic_contour = True

    def BetaFunction(self):
        def fn(x):
            return np.real(special.iv(0, np.sqrt((x + 0j)**2 - 1)) / special.iv(0, x + 0j)) - np.exp(-self.u/2)
        betas = fsolve(fn, 1.0)

        return float(betas[0])

    def Hubbard(self, A, spin=1):
        M = jnp.identity(self.L*self.L) + 0j
        Ab = A.reshape((self.nt, self.L*self.L))
        for t in range(self.nt):
            i, _ = jnp.diag_indices_from(M)
            M = self.h @ jnp.diag(jnp.exp(spin*(1j *
                                  jnp.sin(Ab[t, i]) + self.mu))) @ M
        return jnp.identity(self.L*self.L) + M

    def action(self, A):
        s1, logdet1 = jnp.linalg.slogdet(self.Hubbard(A))
        s2, logdet2 = jnp.linalg.slogdet(self.Hubbard(A, -1))
        return -self.beta * jnp.sum(jnp.cos(A)) - jnp.log(s1) - logdet1 - jnp.log(s2) - logdet2


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
            return np.real(special.iv(1, x + 0j) / (x * special.iv(0, x + 0j))) - self.u
        betas = fsolve(fn, 1.0)

        return float(betas[0])

    def Hubbard(self, A, spin=1):
        M = jnp.identity(self.L*self.L) + 0j
        Ab = A.reshape((self.nt, self.L*self.L))
        for t in range(self.nt):
            i, _ = jnp.diag_indices_from(M)
            M = (-self.kappa*self.hopping + jnp.diag(-1.0 + self.u/2. -
                 spin*(self.mu + 1j*jnp.sin(Ab[t, i])))) @ M
        return jnp.identity(self.L*self.L) + M

    def action(self, A):
        s1, logdet1 = jnp.linalg.slogdet(self.Hubbard(A))
        s2, logdet2 = jnp.linalg.slogdet(self.Hubbard(A, -1))
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
        self.h = {1: self.Hopping.exp_h1(), -1: self.Hopping.exp_h2()}

        self.h1_svd = jnp.linalg.svd(self.h[1])
        self.h2_svd = jnp.linalg.svd(self.h[-1])

        self.dof = self.lattice.dof
        self.periodic_contour = False

    def Hubbard(self, A, spin=1):
        M = jnp.identity(self.L*self.L) + 0j
        Ab = A.reshape((self.nt, self.L*self.L))
        for t in range(self.nt):
            i, _ = jnp.diag_indices_from(M)
            M = self.h[spin] @ jnp.diag(jnp.exp(1j*spin*Ab[t, i])) @ M
        return jnp.identity(self.L*self.L) + M

    def action(self, A):
        s1, logdet1 = jnp.linalg.slogdet(self.Hubbard(A))
        s2, logdet2 = jnp.linalg.slogdet(self.Hubbard(A, -1))
        return jnp.sum(A ** 2) / (2 * self.u) - jnp.log(s1) - logdet1 - jnp.log(s2) - logdet2

    def Hubbard1_svd(self, A):
        fer_mat1 = (jnp.eye(self.lattice.V) + 0j,
                    jnp.ones(self.lattice.V), jnp.eye(self.lattice.V)+0j)

        def update_at_tx(t, x, H):  # don't touch
            H = H.at[x, x].add(
                jnp.exp(1j * A[t * self.lattice.V + x]))
            return H

        def update_at_t(t):  # don't touch. generate B_t
            some = jax.tree_util.Partial(update_at_tx, t)
            temp_mat = jnp.zeros((self.lattice.V, self.lattice.V)) + 0j
            temp_mat = jax.lax.fori_loop(0, self.lattice.V, some, temp_mat)
            return jnp.linalg.svd(temp_mat)

        def multi(t, fer_mat):
            m1 = self.svd_mult(update_at_t(t), fer_mat)
            return self.svd_mult(self.h1_svd, m1)

        fer_mat1 = jax.lax.fori_loop(0, self.lattice.nt, multi, fer_mat1)

        final_svd = jnp.linalg.svd(
            fer_mat1[0].conj().T @ fer_mat1[2].conj().T + jnp.diag(fer_mat1[1]))
        final_u = fer_mat1[0] @ final_svd[0]
        final_d = final_svd[1]
        final_v = final_svd[2] @ fer_mat1[2]

        return final_u, final_d, final_v

    def Hubbard2_svd(self, A):
        fer_mat2 = (jnp.eye(self.lattice.V) + 0j,
                    jnp.ones(self.lattice.V), jnp.eye(self.lattice.V)+0j)

        def update_at_tx(t, x, H):  # don't touch
            H = H.at[x, x].add(
                jnp.exp(-1j * A[t * self.lattice.V + x]))
            return H

        def update_at_t(t):  # don't touch. generate B_t
            some = jax.tree_util.Partial(update_at_tx, t)
            temp_mat = jnp.zeros((self.lattice.V, self.lattice.V)) + 0j
            temp_mat = jax.lax.fori_loop(0, self.lattice.V, some, temp_mat)
            return jnp.linalg.svd(temp_mat)

        def multi(t, fer_mat):
            m1 = self.svd_mult(update_at_t(t), fer_mat)
            return self.svd_mult(self.h2_svd, m1)

        fer_mat2 = jax.lax.fori_loop(0, self.lattice.nt, multi, fer_mat2)

        final_svd = jnp.linalg.svd(
            fer_mat2[0].conj().T @ fer_mat2[2].conj().T + jnp.diag(fer_mat2[1]))
        final_u = fer_mat2[0] @ final_svd[0]
        final_d = final_svd[1]
        final_v = final_svd[2] @ fer_mat2[2]

        return final_u, final_d, final_v

    def action_svd(self, A):
        u1, d1, v1 = self.Hubbard1_svd(A)
        u1_s, u1_logdet = jnp.linalg.slogdet(u1)
        d1_logdet = jnp.sum(jnp.log(d1))
        v1_s, v1_logdet = jnp.linalg.slogdet(v1)
        logdet1 = u1_s + u1_logdet + d1_logdet + v1_s + v1_logdet

        u2, d2, v2 = self.Hubbard2_svd(A)
        u2_s, u2_logdet = jnp.linalg.slogdet(u2)
        d2_logdet = jnp.sum(jnp.log(d2))
        v2_s, v2_logdet = jnp.linalg.slogdet(v2)
        logdet2 = u2_s + u2_logdet + d2_logdet + v2_s + v2_logdet

        return jnp.sum(A ** 2) / (2 * self.u) - logdet1 - logdet2


@dataclass
class ImprovedGaussianModel2(ImprovedModel):
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
        self.h = self.Hopping.exp_h()

        self.dof = self.lattice.dof
        self.periodic_contour = False

    def Hubbard(self, A, spin=1):
        M = jnp.identity(self.L*self.L) + 0j
        Ab = A.reshape((self.nt, self.L*self.L))
        for t in range(self.nt):
            i, _ = jnp.diag_indices_from(M)
            M = self.h @ jnp.diag(jnp.exp(spin*(1j*Ab[t, i]+self.mu))) @ M
        return jnp.identity(self.L*self.L) + M

    def action(self, A):
        s1, logdet1 = jnp.linalg.slogdet(self.Hubbard(A))
        s2, logdet2 = jnp.linalg.slogdet(self.Hubbard(A, -1))
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

    def Hubbard(self, A, spin=1):
        M = jnp.identity(self.L*self.L) + 0j
        Ab = A.reshape((self.nt, self.L*self.L))
        for t in range(self.nt):
            i, _ = jnp.diag_indices_from(M)
            M = (-self.kappa*self.hopping + jnp.diag(-1.0 + self.u/2. -
                 spin*(self.mu + 1j*Ab[t, i]))) @ M
        return jnp.identity(self.L*self.L) + M

    def action(self, A):
        s1, logdet1 = jnp.linalg.slogdet(self.Hubbard(A))
        s2, logdet2 = jnp.linalg.slogdet(self.Hubbard(A, -1))
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

    def Hubbard(self, A, spin=1):
        M = self.hopping_mat + 0j
        Ab = A.reshape((self.nt, self.L*self.L))
        for t in range(self.nt):
            i, _ = jnp.diag_indices_from(M)
            M = jnp.diag(
                -jnp.exp(spin*1j*Ab[t, i])-spin*self.mu) @ self.hopping_inverse @ M
        return jnp.identity(self.L*self.L) + M

    def action(self, A):
        s1, logdet1 = jnp.linalg.slogdet(self.Hubbard(A))
        s2, logdet2 = jnp.linalg.slogdet(self.Hubbard(A, -1))
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
        self.h = {1: self.Hopping.exp_h1(), -1: self.Hopping.exp_h2()}

        self.dof = self.lattice.dof
        self.periodic_contour = False

    def Hubbard(self, A, spin=1):
        M = jnp.identity(self.L*self.L) + 0j
        Ab = A.reshape((2*self.nt, self.L*self.L))
        for t in range(self.nt):
            i, _ = jnp.diag_indices_from(M)
            M = self.h[spin] @ jnp.diag(jnp.exp(1j * spin *
                                                Ab[t, i] + Ab[self.nt + t, i])) @ M
        return jnp.identity(self.L*self.L) + M

    def action(self, A):
        s1, logdet1 = jnp.linalg.slogdet(self.Hubbard(A))
        s2, logdet2 = jnp.linalg.slogdet(self.Hubbard(A, -1))
        return jnp.sum(A[:self.lattice.dof//2] ** 2) / (2 * self.alpha * self.u) + jnp.sum((A[self.lattice.dof//2:] + (1 - self.alpha) * self.u)**2)/(2 * (1 - self.alpha) * self.u) - jnp.log(s1) - logdet1 - jnp.log(s2) - logdet2


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
        self.h = {1: self.Hopping.exp_h1(), -1: self.Hopping.exp_h2()}

        self.dof = self.lattice.dof
        self.periodic_contour = False

    def BetaFunction(self):
        def fn(x):
            return np.real(special.spherical_jn(0, x + 0j)) - np.exp(-self.u/2)
        betas = fsolve(fn, 1.0)

        return float(betas[0])

    def Hubbard(self, A, spin=1):
        M = jnp.identity(self.L*self.L) + 0j
        Ab = A.reshape((self.nt, self.L*self.L))
        for t in range(self.nt):
            i, _ = jnp.diag_indices_from(M)
            M = self.h[spin] @ jnp.diag(jnp.exp(1j *
                                        spin * self.beta * Ab[t, i])) @ M
        return jnp.identity(self.L*self.L) + M

    def action(self, A):
        s1, logdet1 = jnp.linalg.slogdet(self.Hubbard(jnp.tanh(A)))
        s2, logdet2 = jnp.linalg.slogdet(self.Hubbard(jnp.tanh(A), -1))
        return 2 * jnp.sum(jnp.log(jnp.cosh(A))) - jnp.log(s1) - logdet1 - jnp.log(s2) - logdet2


@dataclass
class ImprovedSpinModel(ImprovedModel):
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
        self.h = {1: self.Hopping.exp_h1(), -1: self.Hopping.exp_h2()}

        self.dof = self.lattice.dof
        self.periodic_contour = True

    def BetaFunction(self):
        def fn(x):
            return np.real(special.iv(0, np.sqrt((x + 0j)**2 + 1)) /
                           special.iv(0, x + 0j)) - np.exp(self.u/2)
        betas = fsolve(fn, 1.0)

        return float(betas[0])

    def Hubbard(self, A, spin=1):
        M = jnp.identity(self.L*self.L) + 0j
        Ab = A.reshape((self.nt, self.L*self.L))
        for t in range(self.nt):
            i, _ = jnp.diag_indices_from(M)
            M = self.h[spin] @ jnp.diag(jnp.exp(jnp.sin(Ab[t, i]))) @ M
        return jnp.identity(self.L*self.L) + M

    def action(self, A):
        s1, logdet1 = jnp.linalg.slogdet(self.Hubbard(A))
        s2, logdet2 = jnp.linalg.slogdet(self.Hubbard(A, -1))
        return -self.beta * jnp.sum(jnp.cos(A)) - jnp.sum(jnp.sin(A)) - jnp.log(s1) - logdet1 - jnp.log(s2) - logdet2


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

        self.h1 = self.kappa * self.hopping + \
            (self.mu-self.u)*jnp.identity(self.lattice.V)
        self.h1 = expm(self.h1)

        self.h2 = self.kappa * self.hopping + \
            (-self.mu-self.u)*jnp.identity(self.lattice.V)
        self.h2 = expm(self.h2)

        self.h = {1: self.h1, -1: self.h2}

        self.h1_svd = jnp.linalg.svd(self.h1)
        self.h2_svd = jnp.linalg.svd(self.h2)

        self.dof = self.lattice.dof
        self.periodic_contour = False

    def Hubbard(self, A, spin=1):
        M = jnp.identity(self.L*self.L) + 0j
        Ab = A.reshape((self.nt, self.L*self.L))
        for t in range(self.nt):
            i, _ = jnp.diag_indices_from(M)
            M = self.h[spin] @ jnp.diag(jnp.exp(Ab[t, i])) @ M
        return jnp.identity(self.L*self.L) + M

    def action(self, A):
        s1, logdet1 = jnp.linalg.slogdet(self.Hubbard(A))
        s2, logdet2 = jnp.linalg.slogdet(self.Hubbard(A, -1))
        return jnp.sum(A ** 2) / (2 * self.u) - jnp.log(s1) - logdet1 - jnp.log(s2) - logdet2

    def Hubbard1_svd(self, A):
        fer_mat1 = (jnp.eye(self.lattice.V) + 0j,
                    jnp.ones(self.lattice.V), jnp.eye(self.lattice.V)+0j)

        def update_at_tx(t, x, H):  # don't touch
            H = H.at[x, x].add(
                jnp.exp(A[t * self.lattice.V + x]))
            return H

        def update_at_t(t):  # don't touch. generate B_t
            some = jax.tree_util.Partial(update_at_tx, t)
            temp_mat = jnp.zeros((self.lattice.V, self.lattice.V)) + 0j
            temp_mat = jax.lax.fori_loop(0, self.lattice.V, some, temp_mat)
            return jnp.linalg.svd(temp_mat)

        def multi(t, fer_mat):
            m1 = self.svd_mult(update_at_t(t), fer_mat)
            return self.svd_mult(self.h1_svd, m1)

        fer_mat1 = jax.lax.fori_loop(0, self.lattice.nt, multi, fer_mat1)

        final_svd = jnp.linalg.svd(
            fer_mat1[0].conj().T @ fer_mat1[2].conj().T + jnp.diag(fer_mat1[1]))
        final_u = fer_mat1[0] @ final_svd[0]
        final_d = final_svd[1]
        final_v = final_svd[2] @ fer_mat1[2]

        return final_u, final_d, final_v

    def Hubbard2_svd(self, A):
        fer_mat2 = (jnp.eye(self.lattice.V) + 0j,
                    jnp.ones(self.lattice.V), jnp.eye(self.lattice.V)+0j)

        def update_at_tx(t, x, H):  # don't touch
            H = H.at[x, x].add(
                jnp.exp(A[t * self.lattice.V + x]))
            return H

        def update_at_t(t):  # don't touch. generate B_t
            some = jax.tree_util.Partial(update_at_tx, t)
            temp_mat = jnp.zeros((self.lattice.V, self.lattice.V)) + 0j
            temp_mat = jax.lax.fori_loop(0, self.lattice.V, some, temp_mat)
            return jnp.linalg.svd(temp_mat)

        def multi(t, fer_mat):
            m1 = self.svd_mult(update_at_t(t), fer_mat)
            return self.svd_mult(self.h2_svd, m1)

        fer_mat2 = jax.lax.fori_loop(0, self.lattice.nt, multi, fer_mat2)

        final_svd = jnp.linalg.svd(
            fer_mat2[0].conj().T @ fer_mat2[2].conj().T + jnp.diag(fer_mat2[1]))
        final_u = fer_mat2[0] @ final_svd[0]
        final_d = final_svd[1]
        final_v = final_svd[2] @ fer_mat2[2]

        return final_u, final_d, final_v

    def action_svd(self, A):
        u1, d1, v1 = self.Hubbard1_svd(A)
        u1_s, u1_logdet = jnp.linalg.slogdet(u1)
        d1_logdet = jnp.sum(jnp.log(d1))
        v1_s, v1_logdet = jnp.linalg.slogdet(v1)
        logdet1 = u1_s + u1_logdet + d1_logdet + v1_s + v1_logdet

        u2, d2, v2 = self.Hubbard2_svd(A)
        u2_s, u2_logdet = jnp.linalg.slogdet(u2)
        d2_logdet = jnp.sum(jnp.log(d2))
        v2_s, v2_logdet = jnp.linalg.slogdet(v2)
        logdet2 = u2_s + u2_logdet + d2_logdet + v2_s + v2_logdet

        return jnp.sum(A ** 2) / (2 * self.u) - logdet1 - logdet2


@dataclass
class ImprovedGaussianSpinModel2(ImprovedModel):
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
        self.h = self.Hopping.exp_h()

        self.dof = self.lattice.dof
        self.periodic_contour = False

    def Hubbard(self, A, spin=1):
        M = jnp.identity(self.L*self.L) + 0j
        Ab = A.reshape((self.nt, self.L*self.L))
        for t in range(self.nt):
            i, _ = jnp.diag_indices_from(M)
            M = self.h @ jnp.diag(jnp.exp(Ab[t, i]+spin*self.mu)) @ M
        return jnp.identity(self.L*self.L) + M

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
            return np.real(special.iv(1, x + 0j) / (x * special.iv(0, x + 0j))) - self.u
        betas = fsolve(fn, 1.0)

        return float(betas[0])

    def Hubbard(self, A, spin=1):
        M = jnp.identity(self.L*self.L) + 0j
        Ab = A.reshape((self.nt, self.L*self.L))
        for t in range(self.nt):
            i, _ = jnp.diag_indices_from(M)
            M = (-self.kappa*self.hopping + jnp.diag(-1.0 + self.u/2. -
                 spin*self.mu + jnp.sin(Ab[t, i]))) @ M
        return jnp.identity(self.L*self.L) + M

    def action(self, A):
        s1, logdet1 = jnp.linalg.slogdet(self.Hubbard(A))
        s2, logdet2 = jnp.linalg.slogdet(self.Hubbard(A, -1))
        return -self.beta * jnp.sum(jnp.cos(A)) - jnp.log(s1) - logdet1 - jnp.log(s2) - logdet2
