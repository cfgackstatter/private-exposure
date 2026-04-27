from __future__ import annotations
import logging
from dataclasses import dataclass
import numpy as np
import cvxpy as cp
from sqlmodel import Session

from private_exposure.schemas.optimize import (
    OptimizeRequest, OptimizeResponse, PortfolioOut,
    PortfolioMetricsOut, PositionOut,
)
from private_exposure.services.exposure_builder import build_optimizer_inputs, OptimizerInputs

logger = logging.getLogger(__name__)

_MIN_WEIGHT = 0.005
_RISK_AVERSION = 4.0  # lambda: scales variance penalty in alpha scenarios
_RESIDUAL_PENALTY = 0.1


@dataclass
class _ScenarioConfig:
    label: str
    use_cost: bool
    use_alpha: bool


_SCENARIOS = [
    _ScenarioConfig("Max Exposure", use_cost=False, use_alpha=False),
    _ScenarioConfig("Cost Aware",   use_cost=True,  use_alpha=False),
    _ScenarioConfig("Smart Hedge",  use_cost=True,  use_alpha=True),
]


def run_optimizer(session: Session, req: OptimizeRequest) -> OptimizeResponse:
    inputs = build_optimizer_inputs(
        session, req.keywords,
        max_funds=req.max_funds,
        max_hedges=req.max_hedges,
        flat_borrow_cost=req.borrow_cost,
    )

    warnings: list[str] = []
    if not inputs.fund_labels:
        return OptimizeResponse(
            keywords=req.keywords, portfolios=[],
            warnings=["No funds with target exposure found for these keywords."],
        )

    portfolios = []
    for scenario in _SCENARIOS:
        result = _solve(inputs, req, scenario)
        if result is None:
            warnings.append(f"{scenario.label}: infeasible, skipped.")
            continue
        portfolios.append(_build_portfolio_out(scenario.label, result, inputs, req))
        logger.debug(
            "Scenario: %s | target=%.3f | n_longs=%d | n_shorts=%d",
            scenario.label,
            float(inputs.target_vec @ result.w),
            int((result.w >= _MIN_WEIGHT).sum()),
            int((result.s >= _MIN_WEIGHT).sum()),
        )

    return OptimizeResponse(keywords=req.keywords, portfolios=portfolios, warnings=warnings)


@dataclass
class _SolveResult:
    w: np.ndarray
    s: np.ndarray


def _objective_scale(inp: OptimizerInputs, req: OptimizeRequest) -> float:
    """
    Scale factor to bring cost/signal terms to the same magnitude as target_exp.
    Evaluated at an equal-weight reference portfolio.
    Returns scale such that: normalised_term = term / scale
    """
    n_f = len(inp.fund_labels)
    n_h = len(inp.hedge_labels)
    if n_f == 0:
        return 1.0

    w_ref = np.full(n_f, 1.0 / n_f)
    s_ref = np.full(n_h, req.max_gross_leverage / (2 * max(n_h, 1)))

    target_ref = float(inp.target_vec @ w_ref)

    ann      = 12 / req.holding_period_months
    cost_ref = float(
        inp.fee_vec @ w_ref
        + inp.borrow_vec @ s_ref
        + req.financing_cost * max(w_ref.sum() - 1.0, 0.0)
        + 0.005 * ann * (w_ref.sum() + s_ref.sum())
    )

    alpha_ref = float(inp.alpha_vec @ s_ref) if n_h > 0 else 0.0
    var_ref   = float(0.5 * s_ref @ inp.cov_matrix @ s_ref) if n_h > 0 else 0.0
    signal_ref = abs(alpha_ref - var_ref)

    # representative magnitude of the secondary terms
    secondary = max(cost_ref, signal_ref, 1e-10)
    return secondary / max(target_ref, 1e-10)


def _solve(inp: OptimizerInputs, req: OptimizeRequest, scenario: _ScenarioConfig) -> _SolveResult | None:
    n_f = len(inp.fund_labels)
    n_h = len(inp.hedge_labels)

    w = cp.Variable(n_f, nonneg=True, name="w")
    s = cp.Variable(n_h, nonneg=True, name="s")
    c = cp.Variable(nonneg=True, name="cash")

    target_exp = inp.target_vec @ w
    residual   = inp.hedge_matrix @ w - s
    gross_lev  = cp.sum(w) + cp.sum(s)
    fund_cap   = req.max_fund_weight if req.max_fund_weight is not None else req.max_gross_leverage

    ann   = 12 / req.holding_period_months
    scale = _objective_scale(inp, req)

    total_cost = (
        inp.fee_vec @ w
        + inp.borrow_vec @ s
        + req.financing_cost * cp.maximum(cp.sum(w) - 1.0, 0.0)
        + 0.005 * ann * gross_lev
    )

    cost_term       = (total_cost / scale) if scenario.use_cost else 0.0
    residual_penalty = _RESIDUAL_PENALTY * cp.norm1(residual)  # weight space, no scaling

    if scenario.use_alpha and n_h > 0:
        alpha_term    = -(inp.alpha_vec @ s)          # reward shorting low-alpha names
        variance_term = 0.5 * _RISK_AVERSION * cp.quad_form(s, inp.cov_matrix)
        signal        = (alpha_term - variance_term) / scale
    else:
        signal = 0.0

    objective = cp.Maximize(
        target_exp
        + signal
        - cost_term
        - residual_penalty
    )

    constraints = [
        cp.sum(w) - cp.sum(s) + c == 1.0,
        gross_lev  <= req.max_gross_leverage,
        w          <= fund_cap,
        s          <= req.short_cap,
    ]

    prob = cp.Problem(objective, constraints)
    prob.solve(solver=cp.CLARABEL, warm_start=True)

    if prob.status not in (cp.OPTIMAL, cp.OPTIMAL_INACCURATE):
        return None
    if w.value is None or s.value is None:
        return None

    return _SolveResult(
        w=np.clip(np.array(w.value, dtype=float), 0, None),
        s=np.clip(np.array(s.value, dtype=float), 0, None),
    )


def _active_unhedgeable_names(
    inp: OptimizerInputs, w: np.ndarray, active_funds: set[int], max_names: int = 20,
) -> list[str]:
    """
    Return unhedgeable names only from funds the portfolio actually holds.
    inp.unhedgeable_names_by_fund is a list[set[str]], one entry per fund.
    """
    names: set[str] = set()
    for j in active_funds:
        names.update(inp.unhedgeable_names_by_fund[j])
    return sorted(names)[:max_names]


def _build_portfolio_out(
    label: str, result: _SolveResult, inp: OptimizerInputs, req: OptimizeRequest,
) -> PortfolioOut:
    w, s            = result.w, result.s
    budget          = req.budget
    ann             = 12 / req.holding_period_months

    target_exp      = float(inp.target_vec @ w)
    hedgeable_long  = float((inp.hedge_matrix @ w).sum())
    unhedgeable     = float(inp.unhedgeable_vec @ w)
    non_target_long = hedgeable_long + unhedgeable   # true holding-weight exposure, not fund-weight
    short_total     = float(s.sum())
    net_non_target  = non_target_long - short_total       # residual after hedging
    gross_lev       = float(w.sum() + s.sum())

    carry           = float(
        inp.fee_vec @ w
        + inp.borrow_vec @ s
        + req.financing_cost * max(w.sum() - 1.0, 0.0)
        + 0.005 * ann * (w.sum() + s.sum())
    )

    longs = sorted([
        PositionOut(
            name=inp.fund_names[i], identifier=inp.fund_labels[i],
            weight=float(w[i]), dollar_amount=float(w[i]) * budget,
            annual_cost_pct=float(inp.fee_vec[i]) * 100,
        )
        for i in range(len(inp.fund_labels)) if w[i] >= _MIN_WEIGHT
    ], key=lambda p: p.weight, reverse=True)

    shorts = sorted([
        PositionOut(
            name=inp.hedge_names[j], identifier=inp.hedge_labels[j],
            weight=float(s[j]), dollar_amount=float(s[j]) * budget,
            annual_cost_pct=float(inp.borrow_vec[j]) * 100,
            alpha_pct=round(-float(inp.alpha_vec[j]) * 100, 2),
        )
        for j in range(len(inp.hedge_labels)) if s[j] >= _MIN_WEIGHT
    ], key=lambda p: p.weight, reverse=True)

    active_funds = {j for j in range(len(inp.fund_labels)) if w[j] >= _MIN_WEIGHT}
    unhedgeable_names = _active_unhedgeable_names(inp, w, active_funds)

    return PortfolioOut(
        label=label,
        metrics=PortfolioMetricsOut(
            target_exposure_pct=round(target_exp * 100, 2),
            non_target_long_pct=round(non_target_long * 100, 2),
            short_total_pct=round(short_total * 100, 2),
            net_non_target_pct=round(net_non_target * 100, 2),
            unhedgeable_pct=round(unhedgeable * 100, 2),
            gross_leverage=round(gross_lev, 3),
            annual_carry_cost_pct=round(carry * 100, 2),
            long_count=len(longs),
            short_count=len(shorts),
        ),
        longs=longs,
        shorts=shorts,
        unhedgeable_names=unhedgeable_names,
    )