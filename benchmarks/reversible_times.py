import time

import diffrax as dfx
import equinox as eqx
import jax
import jax.numpy as jnp
import jax.random as jr
import matplotlib.pyplot as plt


jax.config.update("jax_enable_x64", False)


class VectorField(eqx.Module):
    layers: list

    def __init__(self, key):
        key1, key2 = jr.split(key, 2)
        self.layers = [
            eqx.nn.Linear(1000, 1000, use_bias=True, key=key1),
            jnp.tanh,
            eqx.nn.Linear(1000, 1000, use_bias=True, key=key2),
        ]

    def __call__(self, t, y, args):
        for layer in self.layers:
            y = layer(y)
        return y


@eqx.filter_value_and_grad
def grad_loss(y0__term, t1, solver, adjoint):
    y0, term = y0__term
    t0 = 0
    dt0 = 0.01
    max_steps = int((t1 - t0) / dt0)
    ys = dfx.diffeqsolve(
        term,
        solver,
        t0,
        t1,
        dt0,
        y0,
        adjoint=adjoint,
        max_steps=max_steps,
    ).ys

    return jnp.sum(ys**2)


def run(solver, adjoint, t1):
    term = dfx.ODETerm(VectorField(jr.PRNGKey(0)))
    y0 = jr.normal(jr.PRNGKey(0), shape=(100, 1000))

    @jax.vmap
    def _vmap_grad_loss(y0):
        return grad_loss((y0, term), t1, solver, adjoint)

    tic = time.time()
    _vmap_grad_loss(y0)
    toc = time.time()
    runtime = toc - tic

    return runtime


if __name__ == "__main__":
    ts = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
    steps = [int(t1 / 0.01) for t1 in ts]

    base_solver = dfx.MidpointSimple()
    rev_solver = dfx.UReversible(base_solver, coupling_parameter=0.99)

    configs = [
        (rev_solver, dfx.ReversibleAdjoint(), "UReversible + ReversibleAdjoint"),
        (
            base_solver,
            dfx.RecursiveCheckpointAdjoint(),
            "MidpointSimple + RecursiveCheckpoint",
        ),
    ]

    results = {}
    for solver, adjoint, label in configs:
        runtimes = []
        for t1 in ts:
            print(f"t1={t1}")
            # Compile
            run(solver, adjoint, t1)
            # Run
            runtime = run(solver, adjoint, t1)
            runtimes.append(runtime)
        results[label] = runtimes

    plt.figure()
    for label, runtimes in results.items():
        plt.plot(steps, runtimes, marker="o", label=label)
    plt.xlabel("Steps")
    plt.ylabel("Runtime (s)")
    plt.legend()
    plt.savefig("runtimes_f32_vmap.png", dpi=150)
    plt.show()
