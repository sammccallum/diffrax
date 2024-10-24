import jax.numpy as jnp
from diffrax import diffeqsolve, Dopri5, ODETerm, Reversible, SaveAt


def f(t, y, args):
    return -y


term = ODETerm(f)
solver = Reversible(solver=Dopri5(), l=0.999)
y0 = jnp.array([2.0])
saveat = SaveAt(ts=jnp.linspace(0, 1, 100))
sol = diffeqsolve(term, solver, t0=0, t1=1, dt0=0.01, y0=y0, saveat=saveat)
# print(sol.ys)
