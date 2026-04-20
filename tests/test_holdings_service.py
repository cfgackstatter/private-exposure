from __future__ import annotations
from datetime import date
import pytest

from private_exposure.models import FilingRef, FundClass, Holding
from private_exposure.sec.filings_source import FilingsSource
from private_exposure.sec.fund_service import FundService
from private_exposure.sec.fund_source import SecFundSource
from private_exposure.sec.http import SecHttp
from private_exposure.sec.nport_source import NportSource
from private_exposure.services.holdings_service import HoldingsService

_FUND = FundClass(cik=34066, series_id="S000002917", class_id="C000007852",
                  ticker="VTSAX", company_name="Vanguard Index Funds",
                  series_name="Vanguard Total Stock Market Index Fund",
                  class_name="Admiral Shares")

_FILING = FilingRef(accession_no="0000034066-24-000010", cik="0000034066",
                    form_type="NPORT-P", filing_date=date(2024, 3, 29),
                    report_date=date(2024, 1, 31))

_HOLDING = Holding(
    issuer_name="Apple Inc.",
    title="APPLE INC COM",
    cusip="037833100",
    isin="US0378331005",
    ticker="AAPL",
    lei=None,
    other_id=None,
    other_id_desc=None,
    asset_category="EC",
    issuer_type="CORP",
    value_usd=123_456_789.0,
    pct_of_net_assets=4.5,
    quantity=500_000.0,
    quantity_units="NS",
    currency="USD",
    country="US",
    payoff_profile="long",
    fair_value_level="1",
    is_restricted=False,
    is_cash_collateral=None,
    is_non_cash_collateral=None,
    is_loan_by_fund=None,
)


class FakeFunds:
    def find_by_ticker(self, ticker): return _FUND if ticker.upper() == "VTSAX" else None


class FakeFilings:
    def latest_nport(self, cik10, series_id=None, as_of=None):
        if cik10 != _FUND.cik10: return None
        if as_of and _FILING.report_date > as_of: return None
        return _FILING


class FakeNport:
    def get_holdings(self, filing, series_id=None): return [_HOLDING]


def _svc(): return HoldingsService(FakeFunds(), FakeFilings(), FakeNport())


def test_returns_snapshot():
    snap = _svc().get_portfolio("VTSAX")
    assert snap is not None
    assert snap.fund.ticker == "VTSAX"
    assert len(snap.holdings) == 1
    assert snap.holdings[0].issuer_name == "Apple Inc."


def test_unknown_ticker_returns_none():
    assert _svc().get_portfolio("XXXX") is None


def test_as_of_before_filing_returns_none():
    assert _svc().get_portfolio("VTSAX", as_of=date(2020, 1, 1)) is None


def test_as_of_after_filing_returns_snapshot():
    assert _svc().get_portfolio("VTSAX", as_of=date(2025, 1, 1)) is not None


@pytest.mark.integration
def test_live_vtsax():
    http = SecHttp("private-exposure your-email@example.com")
    try:
        svc = HoldingsService(FundService(SecFundSource(http)), FilingsSource(http), NportSource(http))
        snap = svc.get_portfolio("VTSAX")
        assert snap and len(snap.holdings) > 0
        assert all(h.issuer_name for h in snap.holdings)
    finally:
        http.close()