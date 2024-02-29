import numpy as np
import jax
import jax.numpy as jnp


def jackknife(xs, ws=None, Bs=50):  # Bs: Block size
    B = len(xs)//Bs  # number of blocks
    if ws is None:  # for reweighting
        ws = xs*0 + 1

    x = np.array(xs[:B*Bs])
    w = np.array(ws[:B*Bs])

    m = sum(x*w)/sum(w)

    # partition
    block = [[B, Bs], list(x.shape[1:])]
    block_f = [i for sublist in block for i in sublist]
    x = x.reshape(block_f)
    w = w.reshape(block_f)

    # jackknife
    vals = [np.mean(np.delete(x, i, axis=0)*np.delete(w, i, axis=0)) /
            np.mean(np.delete(w, i)) for i in range(B)]
    vals = np.array(vals)
    return m, (np.std(vals.real) + 1j*np.std(vals.imag))*np.sqrt(len(vals)-1)


def bin(xs, ws=None, Bs=50):  # Bs: Block size
    B = len(xs)//Bs  # number of blocks
    if ws is None:  # for reweighting
        ws = xs*0 + 1

    x = np.array(xs[:B*Bs])
    w = np.array(ws[:B*Bs])

    m = sum(x*w)/sum(w)

    # partition
    block = [[B, Bs], list(x.shape[1:])]
    block_f = [i for sublist in block for i in sublist]
    x = x.reshape(block_f)
    w = w.reshape(block_f)

    # jackknife
    vals = [np.mean(x[i]*w[i])/np.mean(w[i]) for i in range(B)]
    vals = np.array(vals)
    return m, (np.std(vals.real) + 1j*np.std(vals.imag))/np.sqrt(len(vals)-1)


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
    for n in range(N):
        s = np.random.choice(range(len(x)), len(x))
        ms.append((sum(y[s]) / sum(w[s])))
    ms = np.array(ms)
    return m, np.std(ms.real) + 1j*np.std(ms.imag)

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

