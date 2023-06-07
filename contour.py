#!/usr/bin/env python

# S-K for the anharmonic oscillator, in 0+1

from models import hubbard
from mc import metropolis, replica

import pickle
import sys
import time
from typing import Callable, Sequence

import flax
import flax.linen as nn
import jax
import jax.numpy as jnp
import numpy as np
import optax
# Don't print annoying CPU warning.
jax.config.update('jax_platform_name', 'cpu')
jax.config.update("jax_debug_nans", True)
jax.config.update("jax_debug_infs", True)


class MLP(nn.Module):
    features: Sequence[int]
    kernel_init: Callable = nn.initializers.variance_scaling(
        1e-5, "fan_in", "truncated_normal")
    bias_init: Callable = nn.initializers.zeros

    @nn.compact
    def __call__(self, x):
        for feat in self.features:
            x = nn.Dense(feat, kernel_init=self.kernel_init,
                         bias_init=self.bias_init)(x)
            x = nn.relu(x)
        return x


class Contour(nn.Module):
    volume: int
    features: Sequence[int]

    @nn.compact
    def __call__(self, x):
        V = self.volume
        zeroinit = nn.initializers.zeros
        # u = MLP(self.features)(x)
        v = MLP(self.features)(x)
        # Start from real plane
        # y_r = nn.Dense(V, kernel_init=zeroinit, bias_init=zeroinit)(u)
        y_i = nn.Dense(V, kernel_init=zeroinit, bias_init=zeroinit)(v)
        y_i = 2.0 * (nn.sigmoid(y_i) - 0.5)  # boundary
        return x + 1j*y_i


class PeriodicContour(nn.Module):
    volume: int
    features: Sequence[int]
    width: int

    @nn.compact
    def __call__(self, x):
        ftns = []
        for i in range(1, self.width+1):
            ftns.append(jnp.cos(i*x))
            ftns.append(jnp.sin(i*x))

        V = self.volume
        xs = jnp.concatenate(ftns, axis=-1)
        zeroinit = nn.initializers.zeros
        # u = MLP(self.features)(xs)
        v = MLP(self.features)(xs)
        # y_r = nn.Dense(V, kernel_init=zeroinit, bias_init=zeroinit)(u)
        y_i = nn.Dense(V, kernel_init=zeroinit, bias_init=zeroinit)(v)
        y_i = 2.0 * (nn.sigmoid(y_i) - 0.5)  # boundary
        return x + 1j*y_i


class AffineContour(nn.Module):
    volume: int
    features: Sequence[int]
    even_indices: jnp.ndarray
    odd_indices: jnp.ndarray

    def NewArray(self, aux, n):
        return aux[n]

    @nn.compact
    def __call__(self, x):
        zeroinit = nn.initializers.zeros
        split = jax.tree_util.Partial(self.NewArray, x)
        x_even = jax.lax.map(split, self.even_indices)
        u_even = MLP(self.features)(x_even)
        v_even = MLP(self.features)(x_even)

        u_s = nn.Dense(1, kernel_init=zeroinit, bias_init=zeroinit)(u_even)
        u_t = nn.Dense(1, kernel_init=zeroinit, bias_init=zeroinit)(v_even)

        y = x+0j

        def update_at_i(i, z):
            z = z.at[self.odd_indices[i]].add(
                1j*(u_s[0] * x[self.odd_indices[i]] + u_t[0]))
            return z

        y = jax.lax.fori_loop(0, self.volume // 2, update_at_i, y)

        return y


class PeriodicAffineContour(nn.Module):
    volume: int
    features: Sequence[int]
    width: int
    even_indices: jnp.ndarray
    odd_indices: jnp.ndarray

    def NewArray(self, aux, n):
        return aux[n]

    @nn.compact
    def __call__(self, x):
        zeroinit = nn.initializers.zeros
        split = jax.tree_util.Partial(self.NewArray, x)
        x_even = jax.lax.map(split, self.even_indices)

        ftns = []
        for i in range(1, self.width+1):
            ftns.append(jnp.cos(i*x_even))
            ftns.append(jnp.sin(i*x_even))

        xs = jnp.concatenate(ftns, axis=-1)

        u_even = MLP(self.features)(xs)
        v_even = MLP(self.features)(xs)

        u_s = nn.Dense(1, kernel_init=zeroinit, bias_init=zeroinit)(u_even)
        u_t = nn.Dense(1, kernel_init=zeroinit, bias_init=zeroinit)(v_even)

        y = x+0j

        def update_at_i(i, z):
            z = z.at[self.odd_indices[i]].add(
                1j*(u_s[0] * x[self.odd_indices[i]] + u_t[0]))
            return z

        y = jax.lax.fori_loop(0, self.volume // 2, update_at_i, y)

        return y


class NearestNeighborAffineContour(nn.Module):
    volume: int
    features: Sequence[int]
    even_indices: jnp.ndarray
    odd_indices: jnp.ndarray
    indftn: Callable[[int], jnp.ndarray]

    def NewArray(self, aux, n):
        return aux[n]

    @nn.compact
    def __call__(self, x):
        zeroinit = nn.initializers.zeros
        split = jax.tree_util.Partial(self.NewArray, x)

        y = x+0j

        def update_at_i(i, z):
            nn = self.indftn(self.even_indices[i])
            x_nn = jax.lax.map(split, nn)
            u_nn = MLP(self.features)(x_nn)
            v_nn = MLP(self.features)(x_nn)

            u_s = nn.Dense(1, kernel_init=zeroinit, bias_init=zeroinit)(u_nn)
            u_t = nn.Dense(1, kernel_init=zeroinit, bias_init=zeroinit)(v_nn)
            z = z.at[self.odd_indices[i]].add(
                1j*(u_s[0] * z[self.odd_indices[i]] + u_t[0]))
            return z

        y = jax.lax.fori_loop(0, self.volume // 2, update_at_i, y)

        return y


class RealContour(nn.Module):
    def __call__(self, x):
        return x


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser(
        description="Train contour",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
        fromfile_prefix_chars='@')
    parser.add_argument('model', type=str, help="model filename")
    parser.add_argument('contour', type=str, help="contour filename")
    parser.add_argument('-R', '--real', action='store_true',
                        help="output the real plane")
    parser.add_argument('-i', '--init', action='store_true',
                        help="re-initialize even if contour already exists")
    parser.add_argument('-f', '--from', dest='fromfile',
                        type=str, help="initialize from other file")
    parser.add_argument('-l', '--layers', type=int, default=0,
                        help='number of (hidden) layers')
    parser.add_argument('-w', '--width', type=int, default=1,
                        help='width (scaling)')
    parser.add_argument('-r', '--replica', action='store_true',
                        help="use replica exchange")
    parser.add_argument('-nrep', '--nreplicas', type=int, default=30,
                        help="number of replicas (with -r)")
    parser.add_argument('-maxh', '--max-hbar', type=float,
                        default=10., help="maximum hbar (with -r)")
    parser.add_argument('--seed', type=int, default=0, help="random seed")
    parser.add_argument('--seed-time', action='store_true',
                        help="seed PRNG with current time")
    parser.add_argument('-lr', '--learningrate', type=float,
                        default=1e-4, help="learning rate")
    parser.add_argument('-S', '--skip', default=30, type=int,
                        help='number of steps to skip')
    parser.add_argument('-A', '--affine', action='store_true',
                        help='use affine coupling layer')
    parser.add_argument('-nnA', '--nnaffine', action='store_true',
                        help='use nearest neighbor affine coupling layer')
    parser.add_argument('-N', '--nstochastic', default=1,
                        type=int, help="number of samples to estimate gradient")
    parser.add_argument('-T', '--thermalize', default=0,
                        type=int, help="number of MC steps (* d.o.f) to thermalize")
    parser.add_argument('-Nt', '--tsteps', default=10000000,
                        type=int, help="number of training")
    parser.add_argument(
        '-o',  '--optimizer', choices=['adam', 'sgd', 'yogi'], default='adam', help='optimizer to use')
    parser.add_argument('-s', '--schedule', action='store_true',
                        help="scheduled learning rate")
    parser.add_argument('-C', '--care', type=float, default=1,
                        help='scaling for learning schedule')
    parser.add_argument('--b1', type=float, default=0.9,
                        help="b1 parameter for adam")
    parser.add_argument('--b2', type=float, default=0.999,
                        help="b2 parameter for adam")
    parser.add_argument(
        '--weight', type=str, default='jnp.ones(len(grads))', help="weight for gradients")

    args = parser.parse_args()

    seed = args.seed
    if args.seed_time:
        seed = time.time_ns()
    key = jax.random.PRNGKey(seed)

    with open(args.model, 'rb') as f:
        model = eval(f.read())
    V = model.dof

    skip = args.skip
    if args.skip == 30:
        skip = V

    if args.affine or args.nnaffine:
        even_indices = model.lattice.even()
        odd_indices = model.lattice.odd()

        @jax.jit
        def Seff(x, p):
            j = jax.jacfwd(lambda y: contour.apply(p, y))(x)
            logdet = jnp.log(j.diagonal().prod())
            xt = contour.apply(p, x)
            Seff = model.action(xt) - logdet
            return Seff

    else:
        @jax.jit
        def Seff(x, p):
            j = jax.jacfwd(lambda y: contour.apply(p, y))(x)
            s, logdet = jnp.linalg.slogdet(j)
            xt = contour.apply(p, x)
            Seff = model.action(xt) - jnp.log(s) - logdet
            return Seff

    if args.nnaffine:
        indftn = model.lattice.nearestneighbor

    contour_ikey, chain_key = jax.random.split(key, 2)

    if args.real:
        # Output real plane and quit
        contour = RealContour()
        contour_params = contour.init(contour_ikey, jnp.zeros(V))
        with open(args.contour, 'wb') as f:
            pickle.dump((contour, contour_params), f)
        sys.exit(0)

    loaded = False
    if not args.init and not args.fromfile:
        try:
            with open(args.contour, 'rb') as f:
                contour, contour_params = pickle.load(f)
            loaded = True
        except FileNotFoundError:
            pass
    if args.fromfile:
        with open(args.fromfile, 'rb') as f:
            contour, contour_params = pickle.load(f)
        loaded = True
    if not loaded:
        if model.periodic_contour:
            if args.affine:
                contour = PeriodicAffineContour(
                    V, [args.width*V] * args.layers, args.width, even_indices, odd_indices)
            else:
                contour = PeriodicContour(
                    V, [args.width*V] * args.layers, args.width)
        else:
            if args.affine:
                contour = AffineContour(
                    V, [args.width*V] * args.layers, even_indices, odd_indices)
            elif args.nnaffine:
                contour = NearestNeighborAffineContour(
                    V, [args.width*4] * args.layers, even_indices, odd_indices, indftn)
            else:
                contour = Contour(V, [args.width*V]*args.layers)
        contour_params = contour.init(contour_ikey, jnp.zeros(V))
    # setup metropolis
    if args.replica:
        chain = replica.ReplicaExchange(lambda x: Seff(x, contour_params), jnp.zeros(
            V), chain_key, max_hbar=args.max_hbar, Nreplicas=args.nreplicas)
    else:
        chain = metropolis.Chain(lambda x: Seff(
            x, contour_params).real, jnp.zeros(V), chain_key)

    Seff_grad = jax.jit(jax.grad(lambda y, p: -Seff(y, p).real, argnums=1))

    if args.schedule:
        sched = optax.exponential_decay(
            init_value=args.learningrate,
            transition_steps=int(args.care*1000),
            decay_rate=0.99,
            end_value=2e-5)
    else:
        sched = optax.constant_schedule(args.learningrate)
    opt = getattr(optax, args.optimizer)(sched, args.b1, args.b2)
    opt_state = opt.init(contour_params)
    opt_update_jit = jax.jit(opt.update)

    def save():
        with open(args.contour, 'wb') as f:
            pickle.dump((contour, contour_params), f)

    def Grad_Mean(grads, weight):
        """
        Params:
            grads: Gradients
            weight: Weights
        """
        grads_w = [jax.tree_util.tree_map(
            lambda x: w*x, g) for w, g in zip(weight, grads)]
        w_mean = jnp.mean(weight)
        grad_mean = jax.tree_util.tree_map(
            lambda *x: jnp.mean(jnp.array(x), axis=0)/w_mean, *grads_w)
        return grad_mean

    def bootstrap(xs, ws=None, N=100, Bs=50):
        if Bs > len(xs):
            Bs = len(xs)
        B = len(xs)//Bs
        if ws is None:
            ws = xs*0 + 1
        # Block
        x, w = [], []
        for i in range(Bs):
            x.append(sum(xs[i*B:i*B+B]*ws[i*B:i*B+B])/sum(ws[i*B:i*B+B]))
            w.append(sum(ws[i*B:i*B+B]))
        x = np.array(x)
        w = np.array(w)
        # Regular bootstrap
        y = x * w
        m = (sum(y) / sum(w))
        ms = []
        for _ in range(N):
            s = np.random.choice(range(len(x)), len(x))
            ms.append((sum(y[s]) / sum(w[s])))
        ms = np.array(ms)
        return m, np.std(ms.real) + 1j*np.std(ms.imag)

    steps = int(10000 / args.nstochastic)
    phases = [0] * steps * args.nstochastic
    acts = [0] * steps * args.nstochastic
    grads = [0] * args.nstochastic
    weight = eval(args.weight)

    chain.calibrate()
    chain.step(N=args.thermalize*V)
    try:
        for t in range(args.tsteps):
            for s in range(steps):
                chain.calibrate()
                for l in range(args.nstochastic):
                    chain.step(N=skip)
                    grads[l] = Seff_grad(chain.x, contour_params)
                    acts[s*args.nstochastic+l] = Seff(chain.x, contour_params)
                    phases[s*args.nstochastic +
                           l] = jnp.exp(-1j*acts[s*args.nstochastic+l].imag)
                grad = Grad_Mean(grads, weight)
                updates, opt_state = opt_update_jit(grad, opt_state)
                contour_params = optax.apply_updates(contour_params, updates)

            # tracking the size of gradient
            grad_abs = 0.
            for i in range(1):
                grad_abs += np.linalg.norm(grad['params']
                                           ['Dense_'+str(i)]['kernel'])
                grad_abs += np.linalg.norm(grad['params']
                                           ['Dense_'+str(i)]['bias'])

                for j in range(args.layers):
                    grad_abs += np.linalg.norm(grad['params']
                                               ['MLP_'+str(i)]['Dense_'+str(j)]['kernel'])
                    grad_abs += np.linalg.norm(grad['params']
                                               ['MLP_'+str(i)]['Dense_'+str(j)]['bias'])

            print(f'{np.mean(phases).real} {np.abs(np.mean(phases))} {bootstrap(np.array(phases))} ({np.mean(np.abs(chain.x))} {np.real(np.mean(acts))} {np.mean(acts)} {grad_abs} {chain.acceptance_rate()})', flush=True)

            save()

    except KeyboardInterrupt:
        print()
        save()
