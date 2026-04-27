from __future__ import annotations
from unittest.mock import MagicMock, patch
from private_exposure.db.models import Fund, Filing, Holding
from private_exposure.services.exposure_builder import (
    _is_hedgeable, _matches_keywords, _holding_identifier, build_optimizer_inputs,
)
from datetime import date


def _holding(
    id: int = 1,
    filing_id: int = 1,
    issuer_name: str = "Test Corp",
    title: str | None = None,
    cusip: str | None = None,
    isin: str | None = None,
    ticker: str | None = None,
    lei: str | None = None,
    other_id: str | None = None,
    other_id_desc: str | None = None,
    asset_category: str | None = "EC",
    issuer_type: str | None = "CORP",
    value_usd: float = 1000.0,
    pct_of_net_assets: float = 1.0,
    quantity: float = 10.0,
    quantity_units: str | None = "NS",
    currency: str | None = "USD",
    country: str | None = "US",
    payoff_profile: str | None = "long",
    fair_value_level: str | None = "1",
    is_restricted: bool | None = False,
    is_cash_collateral: bool | None = False,
    is_non_cash_collateral: bool | None = False,
    is_loan_by_fund: bool | None = False,
    maturity_date: date | None = None,
    coupon_rate: float | None = None,
    is_default: bool | None = None,
    derivative_category: str | None = None,
    notional_amount: float | None = None,
) -> Holding:
    return Holding(
        id=id, filing_id=filing_id, issuer_name=issuer_name,
        title=title, cusip=cusip, isin=isin, ticker=ticker, lei=lei,
        other_id=other_id, other_id_desc=other_id_desc,
        asset_category=asset_category, issuer_type=issuer_type,
        value_usd=value_usd, pct_of_net_assets=pct_of_net_assets,
        quantity=quantity, quantity_units=quantity_units, currency=currency,
        country=country, payoff_profile=payoff_profile,
        fair_value_level=fair_value_level, is_restricted=is_restricted,
        is_cash_collateral=is_cash_collateral,
        is_non_cash_collateral=is_non_cash_collateral,
        is_loan_by_fund=is_loan_by_fund, maturity_date=maturity_date,
        coupon_rate=coupon_rate, is_default=is_default,
        derivative_category=derivative_category, notional_amount=notional_amount,
    )


# ── _is_hedgeable ─────────────────────────────────────────────────────────────

def test_hedgeable_equity_with_ticker():
    assert _is_hedgeable(_holding(ticker="AAPL", asset_category="EC"))


def test_hedgeable_equity_with_isin_only():
    assert _is_hedgeable(_holding(isin="US0378331005", asset_category="EC"))


def test_not_hedgeable_no_identifier():
    assert not _is_hedgeable(_holding(ticker=None, isin=None, asset_category="EC"))


def test_not_hedgeable_restricted():
    assert not _is_hedgeable(_holding(ticker="AAPL", asset_category="EC", is_restricted=True))


def test_not_hedgeable_cash_collateral():
    assert not _is_hedgeable(_holding(ticker="AAPL", asset_category="EC", is_cash_collateral=True))


def test_not_hedgeable_bond_category():
    assert not _is_hedgeable(_holding(ticker="AAPL", asset_category="DBT"))


def test_not_hedgeable_private():
    assert not _is_hedgeable(_holding(ticker=None, isin=None, asset_category="PF"))


# ── _matches_keywords ─────────────────────────────────────────────────────────

def test_matches_title_case_insensitive():
    h = _holding(title="SpaceX Holdings Inc", issuer_name="other")
    assert _matches_keywords(h, ["spacex"])


def test_matches_issuer_name_fallback():
    h = _holding(title=None, issuer_name="SpaceX LLC")
    assert _matches_keywords(h, ["spacex"])


def test_no_match():
    h = _holding(title="Apple Inc", issuer_name="Apple Inc")
    assert not _matches_keywords(h, ["spacex"])


def test_matches_any_keyword():
    h = _holding(title="Tesla Motors", issuer_name="Tesla")
    assert _matches_keywords(h, ["spacex", "tesla"])


# ── _holding_identifier ───────────────────────────────────────────────────────

def test_identifier_prefers_ticker():
    assert _holding_identifier(_holding(ticker="aapl", isin="US0378331005")) == "AAPL"


def test_identifier_falls_back_to_isin():
    assert _holding_identifier(_holding(ticker=None, isin="US0378331005")) == "US0378331005"


def test_identifier_empty_when_none():
    assert _holding_identifier(_holding(ticker=None, isin=None, cusip=None)) == ""


# ── build_optimizer_inputs ────────────────────────────────────────────────────

def _make_session(fund, filing, holdings):
    session = MagicMock()

    def exec_side_effect(stmt):
        result = MagicMock()
        # detect which query by inspecting the statement entity
        stmt_str = str(stmt)
        if "funds" in stmt_str and "filings" not in stmt_str and "holdings" not in stmt_str:
            result.all.return_value = [fund]
        elif "filings" in stmt_str:
            result.first.return_value = filing.id
        else:
            result.all.return_value = holdings
        return result

    session.exec.side_effect = exec_side_effect
    return session


def test_build_splits_target_and_hedgeable():
    fund = Fund(id=1, cik=1, series_id="S000000001", class_id="C1",
                ticker="FAKE", company_name="Fake", series_name="Fake Fund",
                class_name="Investor")
    filing = Filing(id=1, fund_id=1, accession_no="0001-24-000001",
                    form_type="NPORT-P", filing_date=date(2024,3,1),
                    report_date=date(2024,1,31))
    holdings = [
        _holding(id=1, filing_id=1, title="SpaceX LLC", value_usd=4000.0,
                 ticker=None, isin=None, is_restricted=True),  # target but unhedgeable
        _holding(id=2, filing_id=1, title="Apple Inc", ticker="AAPL",
                 asset_category="EC", value_usd=3000.0,
                 is_restricted=False, is_cash_collateral=False,
                 is_non_cash_collateral=False, is_loan_by_fund=False),
        _holding(id=3, filing_id=1, title="Microsoft Corp", ticker="MSFT",
                 asset_category="EC", value_usd=3000.0,
                 is_restricted=False, is_cash_collateral=False,
                 is_non_cash_collateral=False, is_loan_by_fund=False),
        _holding(id=4, filing_id=1, title="Private Corp", ticker=None, isin=None,
                asset_category="PF", value_usd=2000.0, is_restricted=False,
                is_cash_collateral=False, is_non_cash_collateral=False, is_loan_by_fund=False),
    ]
    session = _make_session(fund, filing, holdings)
    inp = build_optimizer_inputs(session, keywords=["spacex"], max_funds=5, max_hedges=10)

    assert "FAKE" in inp.fund_labels
    idx = inp.fund_labels.index("FAKE")
    # SpaceX is target (matched keyword) — it goes to target_vec, not unhedgeable_vec
    # unhedgeable_vec tracks non-target holdings that can't be shorted
    assert inp.target_vec[idx] > 0        # SpaceX matches keyword ✓
    assert inp.unhedgeable_vec[idx] > 0  # Private Corp is unhedgeable but doesn't match keyword ✓
    assert "AAPL" in inp.hedge_labels or "MSFT" in inp.hedge_labels  # non-target hedgeables found