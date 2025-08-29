from typing import cast

import diffrax
import equinox as eqx
import jax
import jax.numpy as jnp
import jax.random as jr
import jax.tree_util as jtu
from jaxtyping import Array


jax.config.update("jax_enable_x64", True)


def _no_nan(x):
    if eqx.is_array(x):
        return x.at[jnp.isnan(x)].set(8.9568)  # arbitrary magic value
    else:
        return x


def tree_allclose(x, y, *, rtol=1e-5, atol=1e-8, equal_nan=False):
    if equal_nan:
        x = jtu.tree_map(_no_nan, x)
        y = jtu.tree_map(_no_nan, y)
    return eqx.tree_equal(x, y, typematch=True, rtol=rtol, atol=atol)


class VectorField(eqx.Module):
    mlp: eqx.nn.MLP

    def __init__(self, in_size, out_size, width_size, depth, key):
        self.mlp = eqx.nn.MLP(in_size, out_size, width_size, depth, key=key)

    def __call__(self, t, y, args):
        return args * self.mlp(y)


@eqx.filter_value_and_grad
def _loss(y0__args__term, solver, saveat, adjoint, stepsize_controller, dual_y0):
    y0, args, term = y0__args__term

    sol = diffrax.diffeqsolve(
        term,
        solver,
        t0=0,
        t1=5,
        dt0=0.01,
        y0=y0,
        args=args,
        saveat=saveat,
        max_steps=4096,
        adjoint=adjoint,
        stepsize_controller=stepsize_controller,
    )

    y1 = sol.ys
    return jnp.sum(cast(Array, y1))


def _compare_grads(
    y0__args__term, base_solver, solver, saveat, stepsize_controller, dual_y0
):
    loss, grads_base = _loss(
        y0__args__term,
        base_solver,
        saveat,
        adjoint=diffrax.RecursiveCheckpointAdjoint(),
        stepsize_controller=stepsize_controller,
        dual_y0=dual_y0,
    )
    loss, grads_reversible = _loss(
        y0__args__term,
        solver,
        saveat,
        adjoint=diffrax.ReversibleAdjoint(),
        stepsize_controller=stepsize_controller,
        dual_y0=dual_y0,
    )
    result = tree_allclose(grads_base, grads_reversible, atol=1e-5)
    print(f"Result: {result}")


if __name__ == "__main__":
    n = 10
    y0 = jnp.linspace(1, 10, num=n)
    key = jr.PRNGKey(10)
    f = VectorField(n, n, n, depth=4, key=key)
    terms = diffrax.ODETerm(f)
    args = jnp.array([0.5])
    base_solver = diffrax.EES27()
    solver = base_solver
    saveat = diffrax.SaveAt(ts=jnp.linspace(0, 5, 50))
    stepsize_controller = diffrax.ConstantStepSize()

    _compare_grads(
        (y0, args, terms),
        base_solver,
        solver,
        saveat,
        stepsize_controller,
        dual_y0=False,
    )
