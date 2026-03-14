from collections.abc import Callable
from typing import ClassVar, TypeAlias

import jax.numpy as jnp
from equinox.internal import ω

from .._custom_types import Args, BoolScalarLike, DenseInfo, RealScalarLike, VF, Y
from .._local_interpolation import LocalLinearInterpolation
from .._solution import RESULTS
from .._term import AbstractTerm
from .base import AbstractAdaptiveSolver, AbstractItoSolver, AbstractSolver


_ErrorEstimate: TypeAlias = None
_SolverState: TypeAlias = None


class MidpointSimple(AbstractItoSolver):
    term_structure: ClassVar = AbstractTerm
    interpolation_cls: ClassVar[Callable[..., LocalLinearInterpolation]] = (
        LocalLinearInterpolation
    )

    def order(self, terms):
        return 2

    def strong_order(self, terms):
        return 0.5

    def init(
        self,
        terms: AbstractTerm,
        t0: RealScalarLike,
        t1: RealScalarLike,
        y0: Y,
        args: Args,
    ) -> _SolverState:
        return None

    def step(
        self,
        terms: AbstractTerm,
        t0: RealScalarLike,
        t1: RealScalarLike,
        y0: Y,
        args: Args,
        solver_state: _SolverState,
        made_jump: BoolScalarLike,
    ) -> tuple[Y, _ErrorEstimate, DenseInfo, _SolverState, RESULTS]:
        del solver_state, made_jump
        thalf = t0 + 0.5 * (t1 - t0)
        control_half = terms.contr(t0, thalf)
        control = terms.contr(t0, t1)
        yhalf = (y0**ω + terms.vf_prod(t0, y0, args, control_half) ** ω).ω
        y1 = (y0**ω + terms.vf_prod(thalf, yhalf, args, control) ** ω).ω
        dense_info = dict(y0=y0, y1=y1)
        return y1, None, dense_info, None, RESULTS.successful

    def func(
        self,
        terms: AbstractTerm,
        t0: RealScalarLike,
        y0: Y,
        args: Args,
    ) -> VF:
        return terms.vf(t0, y0, args)


class RK3Simple(AbstractSolver):
    term_structure: ClassVar = AbstractTerm
    interpolation_cls: ClassVar[Callable[..., LocalLinearInterpolation]] = (
        LocalLinearInterpolation
    )

    def order(self, terms):
        return 3

    def strong_order(self, terms):
        return 0.5

    def init(
        self,
        terms: AbstractTerm,
        t0: RealScalarLike,
        t1: RealScalarLike,
        y0: Y,
        args: Args,
    ) -> _SolverState:
        return None

    def step(
        self,
        terms: AbstractTerm,
        t0: RealScalarLike,
        t1: RealScalarLike,
        y0: Y,
        args: Args,
        solver_state: _SolverState,
        made_jump: BoolScalarLike,
    ) -> tuple[Y, _ErrorEstimate, DenseInfo, _SolverState, RESULTS]:
        del solver_state, made_jump
        control = terms.contr(t0, t1)
        k1 = terms.vf(t0, y0, args)

        _t0 = t0 + 0.5 * (t1 - t0)
        _control = terms.contr(t0, _t0)
        _y0 = (y0**ω + terms.prod(k1, _control) ** ω).ω
        k2 = terms.vf(_t0, _y0, args)

        _t0 = t0 + 0.75 * (t1 - t0)
        _control = terms.contr(t0, _t0)
        _y0 = (y0**ω + terms.prod(k2, _control) ** ω).ω
        k3 = terms.vf(_t0, _y0, args)

        ks = ((2 / 9) * k1**ω + (1 / 3) * k2**ω + (4 / 9) * k3**ω).ω
        y1 = (y0**ω + terms.prod(ks, control) ** ω).ω
        dense_info = dict(y0=y0, y1=y1)
        return y1, None, dense_info, None, RESULTS.successful

    def func(
        self,
        terms: AbstractTerm,
        t0: RealScalarLike,
        y0: Y,
        args: Args,
    ) -> VF:
        return terms.vf(t0, y0, args)


class RK4Simple(AbstractSolver):
    term_structure: ClassVar = AbstractTerm
    interpolation_cls: ClassVar[Callable[..., LocalLinearInterpolation]] = (
        LocalLinearInterpolation
    )

    def order(self, terms):
        return 4

    def strong_order(self, terms):
        return 0.5

    def init(
        self,
        terms: AbstractTerm,
        t0: RealScalarLike,
        t1: RealScalarLike,
        y0: Y,
        args: Args,
    ) -> _SolverState:
        return None

    def step(
        self,
        terms: AbstractTerm,
        t0: RealScalarLike,
        t1: RealScalarLike,
        y0: Y,
        args: Args,
        solver_state: _SolverState,
        made_jump: BoolScalarLike,
    ) -> tuple[Y, _ErrorEstimate, DenseInfo, _SolverState, RESULTS]:
        del solver_state, made_jump
        control = terms.contr(t0, t1)
        k1 = terms.vf(t0, y0, args)

        _t0 = t0 + 0.5 * (t1 - t0)
        _control = terms.contr(t0, _t0)
        _y0 = (y0**ω + terms.prod(k1, _control) ** ω).ω
        k2 = terms.vf(_t0, _y0, args)

        _t0 = t0 + 0.5 * (t1 - t0)
        _control = terms.contr(t0, _t0)
        _y0 = (y0**ω + terms.prod(k2, _control) ** ω).ω
        k3 = terms.vf(_t0, _y0, args)

        _y0 = (y0**ω + terms.prod(k3, control) ** ω).ω
        k4 = terms.vf(t1, _y0, args)
        ks = ((1 / 6) * k1**ω + (1 / 3) * k2**ω + (1 / 3) * k3**ω + (1 / 6) * k4**ω).ω
        y1 = (y0**ω + terms.prod(ks, control) ** ω).ω
        dense_info = dict(y0=y0, y1=y1)
        return y1, None, dense_info, None, RESULTS.successful

    def func(
        self,
        terms: AbstractTerm,
        t0: RealScalarLike,
        y0: Y,
        args: Args,
    ) -> VF:
        return terms.vf(t0, y0, args)


class Bosh3Simple(AbstractAdaptiveSolver):
    term_structure: ClassVar = AbstractTerm
    interpolation_cls: ClassVar[Callable[..., LocalLinearInterpolation]] = (
        LocalLinearInterpolation
    )

    def order(self, terms):
        return 3

    def strong_order(self, terms):
        return 0.5

    def init(
        self,
        terms: AbstractTerm,
        t0: RealScalarLike,
        t1: RealScalarLike,
        y0: Y,
        args: Args,
    ) -> _SolverState:
        return None

    def step(
        self,
        terms: AbstractTerm,
        t0: RealScalarLike,
        t1: RealScalarLike,
        y0: Y,
        args: Args,
        solver_state: _SolverState,
        made_jump: BoolScalarLike,
    ) -> tuple[Y, _ErrorEstimate, DenseInfo, _SolverState, RESULTS]:
        del solver_state, made_jump
        control = terms.contr(t0, t1)
        k1 = terms.vf(t0, y0, args)

        _t0 = t0 + 0.5 * (t1 - t0)
        _control = terms.contr(t0, _t0)
        _y0 = (y0**ω + terms.prod(k1, _control) ** ω).ω
        k2 = terms.vf(_t0, _y0, args)

        _t0 = t0 + 0.75 * (t1 - t0)
        _control = terms.contr(t0, _t0)
        _y0 = (y0**ω + terms.prod(k2, _control) ** ω).ω
        k3 = terms.vf(_t0, _y0, args)

        ks = ((2 / 9) * k1**ω + (1 / 3) * k2**ω + (4 / 9) * k3**ω).ω
        y1 = (y0**ω + terms.prod(ks, control) ** ω).ω

        # Error estimate
        k4 = terms.vf(t1, y1, args)
        ks_error = (
            (2 / 9 - 7 / 24) * k1**ω
            + (1 / 3 - 1 / 4) * k2**ω
            + (4 / 9 - 1 / 3) * k3**ω
            - (1 / 8) * k4**ω
        ).ω
        error = (terms.prod(ks_error, control) ** ω).call(jnp.abs).ω
        dense_info = dict(y0=y0, y1=y1)
        return y1, error, dense_info, None, RESULTS.successful

    def func(
        self,
        terms: AbstractTerm,
        t0: RealScalarLike,
        y0: Y,
        args: Args,
    ) -> VF:
        return terms.vf(t0, y0, args)
