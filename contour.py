#!/usr/bin/env python

from models import hubbard
from mc import metropolis, replica
import pickle
import sys
import time
from typing import Callable, Sequence
from util import *

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
        2e-0, "fan_in", "truncated_normal")
    bias_init: Callable = nn.initializers.zeros

    @nn.compact
    def __call__(self, x):
        for feat in self.features:
            x = nn.Dense(feat, kernel_init=self.kernel_init,
                         bias_init=self.bias_init)(x)
            x = nn.relu(x)
        return x


class ConstantShift(nn.Module):
    volume: int

    @nn.compact
    def __call__(self, x):
        bias = self.param('bias', nn.initializers.zeros, (self.volume,))
        return x + 1j*bias


class Contour(nn.Module):
    volume: int
    features: Sequence[int]

    @nn.compact
    def __call__(self, x):
        V = self.volume
        zeroinit = nn.initializers.zeros
        u = MLP(self.features)(x)
        v = MLP(self.features)(x)
        # Start from real plane
        y_r = nn.Dense(V, kernel_init=zeroinit, bias_init=zeroinit)(u)
        y_i = nn.Dense(V, kernel_init=zeroinit, bias_init=zeroinit)(v)
        return x + y_r + 1j*y_i


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
        u = MLP(self.features)(xs)
        v = MLP(self.features)(xs)
        y_r = nn.Dense(V, kernel_init=zeroinit, bias_init=zeroinit)(u)
        y_i = nn.Dense(V, kernel_init=zeroinit, bias_init=zeroinit)(v)

        return x + y_r + 1j*y_i


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
    parser.add_argument('-c', '--constant', action='store_true',
                        help="constant shift")
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
    parser.add_argument('--dp', action='store_true',
                        help="turn on double precision")

    args = parser.parse_args()

    seed = args.seed
    if args.seed_time:
        seed = time.time_ns()
    key = jax.random.PRNGKey(seed)

    if args.dp:
        jax.config.update('jax_enable_x64', True)

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

    elif args.constant:
        @jax.jit
        def Seff(x, p):
            xt = contour.apply(p, x)
            Seff = model.action(xt)
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
            if args.constant:
                contour = ConstantShift(V)
            else:
                contour = PeriodicContour(
                    V, [args.width] * args.layers, args.width)
        else:
            if args.constant:
                contour = ConstantShift(V)
            else:
                contour = Contour(V, [args.width]*args.layers)
        contour_params = contour.init(contour_ikey, jnp.zeros(V))
    # setup metropolis
    if args.replica:
        chain = replica.ReplicaExchange(lambda x: Seff(x, contour_params), jnp.zeros(
            V), chain_key, delta=1./jnp.sqrt(V), max_hbar=args.max_hbar, Nreplicas=args.nreplicas)
    else:
        chain = metropolis.Chain(lambda x: Seff(
            x, contour_params).real, jnp.zeros(V), chain_key, delta=1./jnp.sqrt(V))

    Seff_grad = jax.jit(jax.grad(lambda y, p: -Seff(y, p).real, argnums=1))

    if args.schedule:
        sched = optax.exponential_decay(
            init_value=args.learningrate,
            transition_steps=int(args.care),
            decay_rate=0.99,
            end_value=1e-6)
    else:
        sched = optax.constant_schedule(args.learningrate)
    opt = getattr(optax, args.optimizer)(sched, args.b1, args.b2)
    opt_state = opt.init(contour_params)
    opt_update_jit = jax.jit(opt.update)

    def save():
        with open(args.contour, 'wb') as f:
            pickle.dump((contour, contour_params), f)

    steps = int(1000 / args.nstochastic)
    grads = [0] * args.nstochastic
    weight = eval(args.weight)

    # measurement
    phases = [0] * steps * args.nstochastic
    acts = [0] * steps * args.nstochastic

    chain.calibrate()
    chain.step(N=args.thermalize*V)
    try:
        for t in range(args.tsteps):
            for s in range(steps):
                chain.calibrate()
                for l in range(args.nstochastic):
                    chain.step(N=skip)
                    grads[l] = Seff_grad(chain.x, contour_params)

                grad = Grad_Mean(grads, weight)
                updates, opt_state = opt_update_jit(grad, opt_state)
                contour_params = optax.apply_updates(contour_params, updates)

            # tracking the size of gradient
            grad_square = sum((w**2).sum()
                              for w in jax.tree_util.tree_leaves(grad["params"]))

            # measurement once in a while
            for i in range(len(phases)):
                chain.step(N=skip)
                acts[i] = Seff(chain.x, contour_params)
                phases[i] = jnp.exp(-1j*acts[i].imag)

            print(f'{np.mean(phases).real} {np.abs(np.mean(phases))} {jackknife(np.array(phases))} ({np.mean(np.abs(chain.x))} {np.real(np.mean(acts))} {np.mean(acts)} {grad_square} {chain.acceptance_rate()})', flush=True)

            save()

    except KeyboardInterrupt:
        print()
        save()
