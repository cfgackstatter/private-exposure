from __future__ import annotations
from dataclasses import dataclass
import numpy as np
import cvxpy as cp
from sqlmodel import Session

from private_exposure.schemas.optimize import (
    OptimizeRequest, OptimizeResponse, PortfolioOut,
    PortfolioMetricsOut, PositionOut,
)
from private_exposure.services.exposure_builder import build_optimizer_inputs, OptimizerInputs

_FRONTIER_LABELS = ["Conservative", "Balanced", "High Purity", "Max Purity"]
_MIN_WEIGHT = 0.005  # drop positions below 0.5% of equity


def run_optimizer(session: Session, req: OptimizeRequest) -> OptimizeResponse:
    inputs = build_optimizer_inputs(
        session, req.keywords,
        max_funds=req.max_funds,
        max_hedges=req.max_hedges,
        flat_borrow_cost=req.borrow_cost,
    )

    warnings = []
    if not inputs.fund_labels:
        return OptimizeResponse(
            keywords=req.keywords,
            portfolios=[],
            warnings=["No funds with target exposure found for these keywords."],
        )

    # frontier: sweep minimum target exposure from ~30% to max feasible
    max_target = float(np.max(inputs.target_vec))
    min_target = max_target * 0.3
    thresholds = np.linspace(min_target, max_target * 0.95, len(_FRONTIER_LABELS))

    portfolios = []
    for label, tau in zip(_FRONTIER_LABELS, thresholds):
        result = _solve(inputs, req, min_target_exposure=float(tau))
        if result is None:
            warnings.append(f"{label}: infeasible, skipped.")
            continue
        portfolios.append(_build_portfolio_out(label, result, inputs, req))

    return OptimizeResponse(
        keywords=req.keywords,
        portfolios=portfolios,
        warnings=warnings,
    )


# ── internal solver ──────────────────────────────────────────────────────────

@dataclass
class _SolveResult:
    w: np.ndarray   # fund weights
    s: np.ndarray   # short hedge weights


def _solve(
    inp: OptimizerInputs,
    req: OptimizeRequest,
    min_target_exposure: float,
) -> _SolveResult | None:
    n_f = len(inp.fund_labels)
    n_h = len(inp.hedge_labels)

    w = cp.Variable(n_f, nonneg=True, name="w")
    s = cp.Variable(n_h, nonneg=True, name="s")
    c = cp.Variable(nonneg=True, name="cash")

    target_exp = inp.target_vec @ w
    residual = inp.hedge_matrix @ w - s
    gross_lev = cp.sum(w) + cp.sum(s)

    # annualized cost terms
    ann_factor = 12 / req.holding_period_months
    fund_fees = inp.fee_vec @ w
    borrow_cost = inp.borrow_vec @ s
    financing_cost = req.financing_cost * cp.maximum(cp.sum(w) - 1.0, 0.0)
    trading_cost = 0.005 * ann_factor * (cp.sum(w) + cp.sum(s))  # 0.5% one-way tc
    total_cost = fund_fees + borrow_cost + financing_cost + trading_cost

    # objective: maximize target exposure, penalize residual + cost + leverage
    objective = cp.Maximize(
        target_exp
        - 0.5 * cp.norm1(residual)
        - total_cost
        - 0.02 * gross_lev
    )

    constraints = [
        cp.sum(w) - cp.sum(s) + c == 1.0,       # capital budget
        gross_lev <= req.max_gross_leverage,
        w <= req.max_fund_weight,
        s <= req.max_short_weight,
        target_exp >= min_target_exposure,
    ]

    prob = cp.Problem(objective, constraints)
    prob.solve(solver=cp.CLARABEL, warm_start=True)

    if prob.status not in (cp.OPTIMAL, cp.OPTIMAL_INACCURATE):
        return None

    return _SolveResult(
        w=np.clip(w.value, 0, None),
        s=np.clip(s.value, 0, None),
    )


# ── result formatting ─────────────────────────────────────────────────────────

def _build_portfolio_out(
    label: str,
    result: _SolveResult,
    inp: OptimizerInputs,
    req: OptimizeRequest,
) -> PortfolioOut:
    budget = req.budget
    w, s = result.w, result.s

    target_exp = float(inp.target_vec @ w)
    residual = inp.hedge_matrix @ w - s
    unhedgeable = float(inp.unhedgeable_vec @ w)
    gross_lev = float(w.sum() + s.sum())
    net_exp = float(w.sum() - s.sum())

    ann = 12 / req.holding_period_months
    carry = float(
        inp.fee_vec @ w
        + inp.borrow_vec @ s
        + req.financing_cost * max(w.sum() - 1.0, 0.0)
        + 0.005 * ann * (w.sum() + s.sum())
    )

    longs = [
        PositionOut(
            name=inp.fund_names[i],
            identifier=inp.fund_labels[i],
            weight=float(w[i]),
            dollar_amount=float(w[i]) * budget,
            annual_cost_pct=float(inp.fee_vec[i]) * 100,
        )
        for i in range(len(inp.fund_labels))
        if w[i] >= _MIN_WEIGHT
    ]
    longs.sort(key=lambda p: p.weight, reverse=True)

    shorts = [
        PositionOut(
            name=inp.hedge_names[j],
            identifier=inp.hedge_labels[j],
            weight=float(s[j]),
            dollar_amount=float(s[j]) * budget,
            annual_cost_pct=float(inp.borrow_vec[j]) * 100,
        )
        for j in range(len(inp.hedge_labels))
        if s[j] >= _MIN_WEIGHT
    ]
    shorts.sort(key=lambda p: p.weight, reverse=True)

    metrics = PortfolioMetricsOut(
        target_exposure_pct=round(target_exp * 100, 2),
        hedgeable_residual_pct=round(float(np.abs(residual).sum()) * 100, 2),
        unhedgeable_residual_pct=round(unhedgeable * 100, 2),
        gross_leverage=round(gross_lev, 3),
        net_exposure=round(net_exp, 3),
        annual_carry_cost_pct=round(carry * 100, 2),
        long_count=len(longs),
        short_count=len(shorts),
    )

    return PortfolioOut(
        label=label,
        metrics=metrics,
        longs=longs,
        shorts=shorts,
        unhedgeable_names=inp.unhedgeable_names[:20],
    )