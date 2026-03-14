import time
from collections.abc import Callable
from math import nan
from tabnanny import check

import diffrax
import equinox as eqx  # https://github.com/patrick-kidger/equinox
import jax
import jax.numpy as jnp
import matplotlib.pyplot as plt
from jaxtyping import Array, Float  # https://github.com/google/jaxtyping


jax.config.update("jax_enable_x64", True)


# Represents the interval [x0, x_final] discretised into n equally-spaced points.
class SpatialDiscretisation(eqx.Module):
    x0: float = eqx.field(static=True)
    x_final: float = eqx.field(static=True)
    vals: Float[Array, "n"]

    @classmethod
    def discretise_fn(cls, x0: float, x_final: float, n: int, fn: Callable):
        if n < 2:
            raise ValueError("Must discretise [x0, x_final] into at least two points")
        vals = jax.vmap(fn)(jnp.linspace(x0, x_final, n))
        return cls(x0, x_final, vals)

    @property
    def δx(self):
        return (self.x_final - self.x0) / (len(self.vals) - 1)

    def binop(self, other, fn):
        if isinstance(other, SpatialDiscretisation):
            if self.x0 != other.x0 or self.x_final != other.x_final:
                raise ValueError("Mismatched spatial discretisations")
            other = other.vals
        return SpatialDiscretisation(self.x0, self.x_final, fn(self.vals, other))

    def __add__(self, other):
        return self.binop(other, lambda x, y: x + y)

    def __mul__(self, other):
        return self.binop(other, lambda x, y: x * y)

    def __radd__(self, other):
        return self.binop(other, lambda x, y: y + x)

    def __rmul__(self, other):
        return self.binop(other, lambda x, y: y * x)

    def __sub__(self, other):
        return self.binop(other, lambda x, y: x - y)

    def __rsub__(self, other):
        return self.binop(other, lambda x, y: y - x)


def laplacian(y: SpatialDiscretisation) -> SpatialDiscretisation:
    y_next = jnp.roll(y.vals, shift=1)
    y_prev = jnp.roll(y.vals, shift=-1)
    Δy = (y_next - 2 * y.vals + y_prev) / (y.δx**2)
    # Dirichlet boundary condition
    Δy = Δy.at[0].set(0)
    Δy = Δy.at[-1].set(0)
    return SpatialDiscretisation(y.x0, y.x_final, Δy)


# Problem
def vector_field(t, y, args):
    return (1 - y) * laplacian(y)


term = diffrax.ODETerm(vector_field)
ic = lambda x: x**2

# Spatial discretisation
x0 = -1
x_final = 1
n = 50
y0 = SpatialDiscretisation.discretise_fn(x0, x_final, n, ic)

# Temporal discretisation
t0 = 0
t_final = 1
dt = 0.0001
max_steps = int(1 / dt)
print(max_steps)
saveat = diffrax.SaveAt(ts=jnp.linspace(t0, t_final, 50))

stepsize_controller = diffrax.ConstantStepSize()

# Solve with Tsit5
base_solver = diffrax.Tsit5()
solver = diffrax.UReversible(base_solver, coupling_parameter=0.5)
# solver = base_solver
sol = diffrax.diffeqsolve(
    term,
    solver,
    t0,
    t_final,
    dt,
    y0,
    saveat=saveat,
    stepsize_controller=stepsize_controller,
    max_steps=max_steps,
)

plt.figure(figsize=(5, 5))
plt.imshow(
    sol.ys.vals,
    origin="lower",
    extent=(x0, x_final, t0, t_final),
    aspect=(x_final - x0) / (t_final - t0),
    cmap="inferno",
)
plt.xlabel("x")
plt.ylabel("t", rotation=0)
plt.clim(0, 1)
plt.colorbar()
plt.savefig("nonlinear_heat_pde.png", dpi=150, bbox_inches="tight")
plt.close()
print("Saved nonlinear_heat_pde.png")


@eqx.filter_value_and_grad
def grad_loss(y0, solver, adjoint, dt=0.0001):
    max_steps = int(1 / dt)
    sol = diffrax.diffeqsolve(
        term,
        solver,
        t0,
        t_final,
        dt,
        y0,
        adjoint=adjoint,
        saveat=diffrax.SaveAt(t1=True),
        stepsize_controller=stepsize_controller,
        max_steps=max_steps,
    )
    return jnp.mean(sol.ys.vals[0])


def compare_runtimes():
    checkpoint_every = 80
    checkpoints = 10_000 // checkpoint_every

    adjoint = diffrax.RecursiveCheckpointAdjoint(checkpoints)
    loss, grads = grad_loss(y0, base_solver, adjoint)
    tic = time.time()
    loss, grads = grad_loss(y0, base_solver, adjoint)
    toc = time.time()
    print(grads.vals[:10])
    print(f"RecursiveCheckpointAdjoint: {(toc - tic):.2f}s")

    adjoint = diffrax.CheckpointedReversibleAdjoint(checkpoint_every)
    loss, grads = grad_loss(y0, solver, adjoint, dt)
    tic = time.time()
    loss, grads = grad_loss(y0, solver, adjoint, dt)
    toc = time.time()
    print(grads.vals[:10])
    print(f"ReversibleAdjoint: {(toc - tic):.2f}s")


def compare_asymptotics():
    checkpoint_every = 10
    base_solver = diffrax.Tsit5()
    solver = diffrax.UReversible(base_solver, coupling_parameter=0.6)
    adjoint = diffrax.CheckpointedReversibleAdjoint(checkpoint_every)
    dts = [0.0001, 0.00005, 0.00001]
    runtimes = []

    for dt in dts:
        print("Compile grad_loss")
        loss, grads = grad_loss(y0, solver, adjoint, dt)
        print("Run grad_loss")
        tic = time.time()
        loss, grads = grad_loss(y0, solver, adjoint, dt)
        toc = time.time()
        runtimes.append(toc - tic)

        nan_check = jnp.isnan(grads.vals)
        assert not nan_check.any()

    plt.plot(dts, runtimes, ".-")
    plt.savefig("runtimes.png")

