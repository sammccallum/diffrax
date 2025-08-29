from collections.abc import Callable
from math import sqrt
from typing import cast, ClassVar

import numpy as np
from equinox.internal import ω
from jaxtyping import PyTree

from .._custom_types import Args, BoolScalarLike, DenseInfo, RealScalarLike, Y
from .._local_interpolation import (
    ThirdOrderHermitePolynomialInterpolation,
)
from .._solution import RESULTS
from .._solver.base import (
    AbstractReversibleSolver,
)
from .._term import AbstractTerm
from .runge_kutta import AbstractERK, ButcherTableau


ω = cast(Callable, ω)
_SolverState = Y

SQRT2 = sqrt(2)

_ees27_tableau = ButcherTableau(
    a_lower=(
        np.array([(2 - SQRT2) / 3]),
        np.array([(-4 + SQRT2) / 24, (4 + SQRT2) / 8]),
        np.array(
            [(-176 + 145 * SQRT2) / 168, 3 * (8 - 5 * SQRT2) / 56, 3 * (3 - SQRT2) / 7]
        ),
    ),
    b_sol=np.array(
        [
            (5 - 3 * SQRT2) / 14,
            (3 + SQRT2) / 14,
            3 * (-1 + 2 * SQRT2) / 14,
            (9 - 4 * SQRT2) / 14,
        ]
    ),
    b_error=np.array(
        [
            1.0 - (5 - 3 * SQRT2) / 14,
            -(3 + SQRT2) / 14,
            -3 * (-1 + 2 * SQRT2) / 14,
            -(9 - 4 * SQRT2) / 14,
        ]
    ),
    c=np.array([(2 - SQRT2) / 3, (2 + SQRT2) / 6, (4 + SQRT2) / 6]),
)


class EES27(AbstractERK, AbstractReversibleSolver):
    """Explicit and Effectively Symmetric (EES) Runge-Kutta scheme of order 2
    and antisymmetric order 7, with parameter :math:`x = (5-3\\sqrt{2})/14`.
    """

    tableau: ClassVar[ButcherTableau] = _ees27_tableau
    interpolation_cls: ClassVar[
        Callable[..., ThirdOrderHermitePolynomialInterpolation]
    ] = ThirdOrderHermitePolynomialInterpolation.from_k
    # interpolation_cls: ClassVar[Callable[..., LocalLinearInterpolation]] = (
    #     LocalLinearInterpolation
    # )

    def order(self, terms):
        del terms
        return 2

    def antisymmetric_order(self, terms):
        del terms
        return 7

    def backward_step(
        self,
        terms: PyTree[AbstractTerm],
        t0: RealScalarLike,
        t1: RealScalarLike,
        y1: Y,
        args: Args,
        solver_state: _SolverState,
        made_jump: BoolScalarLike,
    ) -> tuple[Y, DenseInfo, _SolverState, RESULTS]:
        y0, _, dense_info, solver_state, result = self.step(
            terms, t1, t0, y1, args, solver_state, made_jump
        )
        return y0, dense_info, solver_state, result
