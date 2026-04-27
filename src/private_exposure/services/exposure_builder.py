from __future__ import annotations
import logging
from dataclasses import dataclass, field
import numpy as np
from sqlmodel import Session, select, col
from private_exposure.db.models import Fund, Filing, Holding as HoldingRow
from private_exposure.api.search import parse_query, matches as search_matches, Token
from private_exposure.services.alpha_service import compute_alphas, compute_covariance, get_signals

logger = logging.getLogger(__name__)

@dataclass
class OptimizerInputs:
    fund_labels: list[str]          # tickers
    fund_names: list[str]           # series names
    hedge_labels: list[str]         # tickers/ISINs of shortable names
    hedge_names: list[str]
    target_vec: np.ndarray          # (n_funds,) target exposure fraction
    hedge_matrix: np.ndarray        # (n_hedges, n_funds) hedgeable exposure matrix
    unhedgeable_vec: np.ndarray     # (n_funds,) unhedgeable residual fraction
    fee_vec: np.ndarray             # (n_funds,) annual expense ratio
    borrow_vec: np.ndarray          # (n_hedges,) annual borrow cost
    alpha_vec: np.ndarray           # (n_hedges,) annualized momentum alpha
    cov_matrix: np.ndarray   # (n_hedges, n_hedges) annualised covariance, PSD-regularised
    unhedgeable_names_by_fund: list[set[str]] = field(default_factory=list)


_HEDGEABLE_CATS = {"EC", "RF"}
_DEFAULT_FUND_FEE = 0.02
_BORROW_COST_QUINTILES = [0.01, 0.02, 0.03, 0.04, 0.05]  # 0.5% to 2.5%
_FALLBACK_VOL = 0.3  # 30% annual vol for unknown tickers, used to assign borrow cost


def _is_hedgeable(h: HoldingRow) -> bool:
    """Only hedgeable if it has a usable identifier — must match _holding_identifier priority."""
    return (
        not h.is_restricted
        and not h.is_cash_collateral
        and not h.is_non_cash_collateral
        and not h.is_loan_by_fund
        and h.asset_category in _HEDGEABLE_CATS
        and bool(h.isin or h.cusip or h.ticker)  # same priority as _holding_identifier
    )


def _vol_based_borrow(hedge_labels: list[str], flat_fallback: float) -> np.ndarray:
    """Assign borrow cost by vol quintile — higher vol → higher borrow cost."""
    signals = get_signals(hedge_labels)
    vols = np.array([signals[t][1] if t in signals else _FALLBACK_VOL for t in hedge_labels])

    n = len(vols)
    if n == 0:
        return np.array([])
    if n == 1:
        return np.array([flat_fallback])

    # rank into quintiles (0-4)
    ranks      = vols.argsort().argsort()                    # rank by vol ascending
    quintiles  = (ranks * 5 // n).clip(0, 4)                # map to [0,4]
    borrow_vec = np.array([_BORROW_COST_QUINTILES[q] for q in quintiles])
    return borrow_vec


def _holding_identifier(h: HoldingRow) -> str:
    """Canonical identifier: ISIN preferred, CUSIP fallback, ticker last."""
    return (h.isin or h.cusip or h.ticker or "").upper().strip()


def _matches_keywords(h: HoldingRow, tokens: list[Token]) -> bool:
    return search_matches(h.issuer_name or "", tokens)


def build_optimizer_inputs(
    session: Session,
    keywords: list[str],
    max_funds: int = 10,
    max_hedges: int = 30,
    flat_borrow_cost: float = 0.05,
) -> OptimizerInputs:
    tokens = parse_query(" OR ".join(keywords))
    funds = session.exec(select(Fund)).all()
    candidates = []

    for fund in funds:
        latest_filing_id = session.exec(
            select(Filing.id)
            .where(Filing.fund_id == fund.id)
            .order_by(col(Filing.report_date).desc())
            .limit(1)
        ).first()
        if latest_filing_id is None:
            continue
        holdings = list(session.exec(
            select(HoldingRow).where(HoldingRow.filing_id == latest_filing_id)
        ).all())
        if not holdings:
            continue
        target = sum(h.pct_of_net_assets / 100.0 for h in holdings
             if h.pct_of_net_assets > 0 and _matches_keywords(h, tokens))
        if target == 0:
            continue
        candidates.append((fund, holdings, target))

    candidates.sort(key=lambda x: x[2], reverse=True)
    candidates = candidates[:max_funds]

    if not candidates:
        return _empty_inputs()

    # build hedge universe: deduplicated, keyed by identifier, max weight across funds
    hedge_universe: dict[str, str] = {}  # ident -> display name
    agg: dict[str, float] = {}           # ident -> max fund weight

    for _, holdings, _ in candidates:
        for h in holdings:
            if _matches_keywords(h, tokens) or not _is_hedgeable(h):
                continue
            ident = _holding_identifier(h)
            if not ident:
                continue
            if ident not in hedge_universe:
                hedge_universe[ident] = h.title or h.issuer_name or ident
                agg[ident] = 0.0
            agg[ident] = max(agg[ident], h.pct_of_net_assets / 100.0)

    top_hedges = sorted(agg, key=lambda k: agg[k], reverse=True)[:max_hedges]
    hedge_labels = top_hedges
    hedge_names = [hedge_universe[k] for k in hedge_labels]

    cov_matrix = compute_covariance(hedge_labels) if hedge_labels else np.empty((0, 0))

    alpha_map = compute_alphas(hedge_labels, cov_matrix=cov_matrix)
    alpha_vec = np.array([alpha_map.get(t, 0.0) for t in hedge_labels])
    
    borrow_vec = _vol_based_borrow(hedge_labels, flat_borrow_cost)

    n_f = len(candidates)
    n_h = len(hedge_labels)
    hedge_idx = {ident: i for i, ident in enumerate(hedge_labels)}

    target_vec      = np.zeros(n_f)
    hedge_matrix    = np.zeros((n_h, n_f))
    unhedgeable_vec = np.zeros(n_f)
    fee_vec = np.array([
        float(f.expense_ratio) if f.expense_ratio else _DEFAULT_FUND_FEE
        for f, _, _ in candidates
    ])

    unhedgeable_names_by_fund: list[set[str]] = [set() for _ in candidates]

    for j, (_, holdings, _) in enumerate(candidates):
        for h in holdings:
            w = h.pct_of_net_assets / 100.0
            if w <= 0:
                continue
            if _matches_keywords(h, tokens):
                target_vec[j] += w
            elif _is_hedgeable(h):
                ident = _holding_identifier(h)
                if ident in hedge_idx:
                    hedge_matrix[hedge_idx[ident], j] += w
                else:
                    unhedgeable_vec[j] += w
                    unhedgeable_names_by_fund[j].add(h.title or h.issuer_name or "unknown")
            else:
                unhedgeable_vec[j] += w
                unhedgeable_names_by_fund[j].add(h.title or h.issuer_name or "unknown")

    return OptimizerInputs(
        fund_labels=[f.ticker for f, _, _ in candidates],
        fund_names=[f.series_name for f, _, _ in candidates],
        hedge_labels=hedge_labels,
        hedge_names=hedge_names,
        target_vec=target_vec,
        hedge_matrix=hedge_matrix,
        unhedgeable_vec=unhedgeable_vec,
        fee_vec=fee_vec,
        borrow_vec=borrow_vec,
        alpha_vec=alpha_vec,
        cov_matrix=cov_matrix,
        unhedgeable_names_by_fund=unhedgeable_names_by_fund,
    )


def _empty_inputs() -> OptimizerInputs:
    empty = np.array([])
    return OptimizerInputs(
        fund_labels=[], fund_names=[], hedge_labels=[], hedge_names=[],
        target_vec=empty, hedge_matrix=np.empty((0, 0)),
        unhedgeable_vec=empty, fee_vec=empty, borrow_vec=empty,
        alpha_vec=empty, cov_matrix=np.empty((0, 0)), unhedgeable_names_by_fund=[],
    )