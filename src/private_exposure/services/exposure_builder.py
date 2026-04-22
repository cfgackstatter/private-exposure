from __future__ import annotations
from dataclasses import dataclass, field
import numpy as np
from sqlmodel import Session, select
from private_exposure.db.models import Fund, Holding as HoldingRow


@dataclass
class OptimizerInputs:
    fund_labels: list[str]          # tickers
    fund_names: list[str]           # series names
    hedge_labels: list[str]         # tickers/ISINs of shortable names
    hedge_names: list[str]
    # target exposure per fund (fraction of fund NAV)
    target_vec: np.ndarray          # shape (n_funds,)
    # hedgeable non-target exposure matrix: A[i,j] = security i exposure in fund j
    hedge_matrix: np.ndarray        # shape (n_hedges, n_funds)
    # unhedgeable residual per fund
    unhedgeable_vec: np.ndarray     # shape (n_funds,)
    # per-fund expense ratio (annual fraction)
    fee_vec: np.ndarray             # shape (n_funds,)
    # per-hedge borrow cost (annual fraction), defaults to flat assumption
    borrow_vec: np.ndarray          # shape (n_hedges,)
    # unhedgeable holding names per fund for display
    unhedgeable_names: list[str] = field(default_factory=list)


_HEDGEABLE_CATS = {"EC", "RF"}  # equity and registered fund are shortable in v1


def _is_hedgeable(h: HoldingRow) -> bool:
    return (
        not h.is_restricted
        and not h.is_cash_collateral
        and not h.is_non_cash_collateral
        and not h.is_loan_by_fund
        and h.asset_category in _HEDGEABLE_CATS
        and (h.ticker or h.isin)
    )


def _holding_identifier(h: HoldingRow) -> str:
    return (h.ticker or h.isin or h.cusip or "").upper()


def _matches_keywords(h: HoldingRow, keywords: list[str]) -> bool:
    name = (h.title or h.issuer_name or "").lower()
    return any(kw.lower() in name for kw in keywords)


def build_optimizer_inputs(
    session: Session,
    keywords: list[str],
    max_funds: int = 10,
    max_hedges: int = 30,
    flat_borrow_cost: float = 0.05,
) -> OptimizerInputs:
    # load candidate funds that have matched holdings
    funds = session.exec(select(Fund)).all()
    candidates: list[tuple[Fund, list[HoldingRow]]] = []

    for fund in funds:
        holdings = session.exec(
            select(HoldingRow).where(HoldingRow.fund_id == fund.id)
        ).all()
        if not holdings:
            continue
        total_value = sum(h.value_usd for h in holdings) or 1.0
        target = sum(h.value_usd for h in holdings if _matches_keywords(h, keywords))
        if target == 0:
            continue
        candidates.append((fund, holdings, target / total_value))

    # sort by target exposure, take top N
    candidates.sort(key=lambda x: x[2], reverse=True)
    candidates = candidates[:max_funds]

    if not candidates:
        return _empty_inputs()

    # build hedge universe: union of hedgeable non-target names across all funds
    hedge_universe: dict[str, str] = {}  # identifier -> display name
    for fund, holdings, _ in candidates:
        for h in holdings:
            if _matches_keywords(h, keywords):
                continue
            if _is_hedgeable(h):
                ident = _holding_identifier(h)
                if ident and ident not in hedge_universe:
                    hedge_universe[ident] = h.title or h.issuer_name or ident

    # limit hedge universe to top names by aggregate cross-fund exposure
    agg: dict[str, float] = {ident: 0.0 for ident in hedge_universe}
    for fund, holdings, _ in candidates:
        total_value = sum(h.value_usd for h in holdings) or 1.0
        for h in holdings:
            if _matches_keywords(h, keywords):
                continue
            ident = _holding_identifier(h)
            if ident in agg:
                agg[ident] += h.value_usd / total_value

    top_hedges = sorted(agg, key=lambda k: agg[k], reverse=True)[:max_hedges]
    hedge_labels = top_hedges
    hedge_names = [hedge_universe[k] for k in hedge_labels]
    n_f = len(candidates)
    n_h = len(hedge_labels)
    hedge_idx = {ident: i for i, ident in enumerate(hedge_labels)}

    target_vec = np.zeros(n_f)
    hedge_matrix = np.zeros((n_h, n_f))
    unhedgeable_vec = np.zeros(n_f)
    fee_vec = np.zeros(n_f)
    unhedgeable_names: set[str] = set()

    for j, (fund, holdings, _) in enumerate(candidates):
        total_value = sum(h.value_usd for h in holdings) or 1.0
        for h in holdings:
            w = h.value_usd / total_value
            if _matches_keywords(h, keywords):
                target_vec[j] += w
            elif _is_hedgeable(h):
                ident = _holding_identifier(h)
                if ident in hedge_idx:
                    hedge_matrix[hedge_idx[ident], j] += w
            else:
                unhedgeable_vec[j] += w
                unhedgeable_names.add(h.title or h.issuer_name or "unknown")

    return OptimizerInputs(
        fund_labels=[f.ticker for f, _, _ in candidates],
        fund_names=[f.series_name for f, _, _ in candidates],
        hedge_labels=hedge_labels,
        hedge_names=hedge_names,
        target_vec=target_vec,
        hedge_matrix=hedge_matrix,
        unhedgeable_vec=unhedgeable_vec,
        fee_vec=fee_vec,
        borrow_vec=np.full(n_h, flat_borrow_cost),
        unhedgeable_names=sorted(unhedgeable_names),
    )


def _empty_inputs() -> OptimizerInputs:
    empty = np.array([])
    return OptimizerInputs(
        fund_labels=[], fund_names=[], hedge_labels=[], hedge_names=[],
        target_vec=empty, hedge_matrix=np.empty((0, 0)),
        unhedgeable_vec=empty, fee_vec=empty, borrow_vec=empty,
    )