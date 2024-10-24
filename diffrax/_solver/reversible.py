from typing import (
    Optional,
    TypeVar,
)

import jax
from jaxtyping import PyTree

from .._custom_types import Args, BoolScalarLike, DenseInfo, RealScalarLike, VF, Y
from .._solution import RESULTS, update_result
from .._term import AbstractTerm
from .base import AbstractAdaptiveSolver, AbstractWrappedSolver


_SolverState = TypeVar("_SolverState")


class Reversible(
    AbstractAdaptiveSolver[_SolverState], AbstractWrappedSolver[_SolverState]
):
    solver: AbstractAdaptiveSolver[_SolverState]
    l: float = 0.999

    @property
    def term_structure(self):
        return self.solver.term_structure

    @property
    def interpolation_cls(self):  # pyright: ignore
        return self.solver.interpolation_cls

    @property
    def term_compatible_contr_kwargs(self):
        return self.solver.term_compatible_contr_kwargs

    def order(self, terms: PyTree[AbstractTerm]) -> Optional[int]:
        return self.solver.order(terms)

    def strong_order(self, terms: PyTree[AbstractTerm]) -> Optional[RealScalarLike]:
        return self.solver.strong_order(terms)

    def init(
        self,
        terms: PyTree[AbstractTerm],
        t0: RealScalarLike,
        t1: RealScalarLike,
        y0: Y,
        args: Args,
    ) -> _SolverState:
        original_solver_init = self.solver.init(terms, t0, t1, y0, args)
        return (original_solver_init, y0)

    def step(
        self,
        terms: PyTree[AbstractTerm],
        t0: RealScalarLike,
        t1: RealScalarLike,
        y0: Y,
        args: Args,
        solver_state: _SolverState,
        made_jump: BoolScalarLike,
    ) -> tuple[Y, Optional[Y], DenseInfo, _SolverState, RESULTS]:
        original_solver_state, z0 = solver_state
        jax.debug.print("{y0}", y0=y0)
        jax.debug.print("{z0}", z0=z0)
        step_z0, _, dense_info, original_solver_state, result1 = self.solver.step(
            terms, t0, t1, z0, args, original_solver_state, made_jump
        )
        y1 = self.l * (y0 - z0) + step_z0

        step_y1, _, _, _, result2 = self.solver.step(
            terms, t1, t0, y1, args, original_solver_state, made_jump
        )
        z1 = y1 + z0 - step_y1

        solver_state = (original_solver_state, z1)
        result = update_result(result1, result2)

        return y1, None, dense_info, solver_state, result

    def func(
        self, terms: PyTree[AbstractTerm], t0: RealScalarLike, y0: Y, args: Args
    ) -> VF:
        return self.solver.func(terms, t0, y0, args)
