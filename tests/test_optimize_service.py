from __future__ import annotations
import numpy as np
import pytest
from unittest.mock import MagicMock, patch
from private_exposure.schemas.optimize import OptimizeRequest
from private_exposure.services.optimize_service import run_optimizer, _solve
from private_exposure.services.exposure_builder import OptimizerInputs


def _simple_inputs(n_funds=2, n_hedges=2) -> OptimizerInputs:
    """Toy exposure matrix: fund 0 is pure target, fund 1 is mixed."""
    return OptimizerInputs(
        fund_labels=["PURE", "MIXED"],
        fund_names=["Pure Fund", "Mixed Fund"],
        hedge_labels=["AAPL", "MSFT"],
        hedge_names=["Apple Inc", "Microsoft Corp"],
        target_vec=np.array([0.80, 0.30]),
        hedge_matrix=np.array([
            [0.10, 0.40],   # AAPL exposure per fund
            [0.10, 0.30],   # MSFT exposure per fund
        ]),
        unhedgeable_vec=np.array([0.00, 0.00]),
        fee_vec=np.array([0.0050, 0.0075]),
        borrow_vec=np.array([0.05, 0.05]),
    )


def _req(
    keywords: list[str] | None = None,
    budget: float = 100_000,
    max_gross_leverage: float = 3.0,
    max_fund_weight: float | None = None,
    max_short_weight: float = 0.10,
    borrow_cost: float = 0.05,
    financing_cost: float = 0.03,
    holding_period_months: int = 12,
    max_funds: int = 10,
    max_hedges: int = 30,
) -> OptimizeRequest:
    return OptimizeRequest(
        keywords=keywords or ["spacex"],
        budget=budget,
        max_gross_leverage=max_gross_leverage,
        max_fund_weight=max_fund_weight,
        max_short_weight=max_short_weight,
        borrow_cost=borrow_cost,
        financing_cost=financing_cost,
        holding_period_months=holding_period_months,
        max_funds=max_funds,
        max_hedges=max_hedges,
    )


# ── _solve ────────────────────────────────────────────────────────────────────

def test_solve_returns_result_for_feasible_problem():
    inp = _simple_inputs()
    result = _solve(inp, _req(), min_target_exposure=0.2)
    assert result is not None
    assert result.w.sum() > 0


def test_solve_respects_leverage_constraint():
    inp = _simple_inputs()
    req = _req(max_gross_leverage=1.5)
    result = _solve(inp, req, min_target_exposure=0.1)
    assert result is not None
    assert result.w.sum() + result.s.sum() <= 1.5 + 1e-4


def test_solve_returns_none_for_infeasible_target():
    inp = _simple_inputs()
    # require 99% target exposure — impossible given the fund mix
    result = _solve(inp, _req(), min_target_exposure=0.99)
    assert result is None


def test_higher_target_threshold_gives_more_target_exposure():
    inp = _simple_inputs()
    req = _req()
    low = _solve(inp, req, min_target_exposure=0.10)
    high = _solve(inp, req, min_target_exposure=0.30)
    assert low is not None and high is not None
    t_low = float(inp.target_vec @ low.w)
    t_high = float(inp.target_vec @ high.w)
    assert t_high >= t_low - 1e-4


def test_short_weights_nonneg():
    inp = _simple_inputs()
    result = _solve(inp, _req(), min_target_exposure=0.2)
    assert result is not None
    assert (result.s >= -1e-6).all()


def test_fund_weights_nonneg():
    inp = _simple_inputs()
    result = _solve(inp, _req(), min_target_exposure=0.2)
    assert result is not None
    assert (result.w >= -1e-6).all()


# ── run_optimizer ─────────────────────────────────────────────────────────────

def test_run_optimizer_returns_multiple_portfolios():
    session = MagicMock()
    with patch(
        "private_exposure.services.optimize_service.build_optimizer_inputs",
        return_value=_simple_inputs(),
    ):
        resp = run_optimizer(session, _req())

    assert len(resp.portfolios) >= 2
    assert all(p.metrics.gross_leverage <= 2.0 + 1e-4 for p in resp.portfolios)


def test_run_optimizer_empty_when_no_funds():
    session = MagicMock()
    empty = OptimizerInputs(
        fund_labels=[], fund_names=[], hedge_labels=[], hedge_names=[],
        target_vec=np.array([]), hedge_matrix=np.empty((0, 0)),
        unhedgeable_vec=np.array([]), fee_vec=np.array([]), borrow_vec=np.array([]),
    )
    with patch(
        "private_exposure.services.optimize_service.build_optimizer_inputs",
        return_value=empty,
    ):
        resp = run_optimizer(session, _req())

    assert resp.portfolios == []
    assert len(resp.warnings) > 0


def test_run_optimizer_labels_are_ordered():
    session = MagicMock()
    with patch(
        "private_exposure.services.optimize_service.build_optimizer_inputs",
        return_value=_simple_inputs(),
    ):
        resp = run_optimizer(session, _req())

    labels = [p.label for p in resp.portfolios]
    assert labels[0] == "Conservative"
    assert labels[-1] == "Max Purity"