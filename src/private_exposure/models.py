from __future__ import annotations

from dataclasses import dataclass, field
from datetime import date
from typing import Protocol


@dataclass(frozen=True, slots=True)
class FundClass:
    cik: int
    series_id: str
    class_id: str
    ticker: str
    company_name: str
    series_name: str
    class_name: str

    @property
    def cik10(self) -> str:
        return f"{self.cik:010d}"


class FundClassReader(Protocol):
    def get_fund_classes(self) -> list[FundClass]: ...
    def find_by_ticker(self, ticker: str) -> FundClass | None: ...
    def find_by_cik(self, cik: int, ticker: str) -> FundClass | None: ...


class FundLookup(Protocol):
    def find_by_ticker(self, ticker: str) -> FundClass | None: ...


@dataclass(frozen=True, slots=True)
class FilingRef:
    accession_no: str
    cik: str
    form_type: str
    filing_date: date
    report_date: date | None


class FilingLookup(Protocol):
    def latest_nport(
        self,
        cik10: str,
        series_id: str | None = None,
        as_of: date | None = None,
    ) -> FilingRef | None: ...


@dataclass(frozen=True, slots=True)
class Holding:
    issuer_name: str
    value_usd: float
    pct_of_net_assets: float
    quantity: float
    title: str | None = None
    cusip: str | None = None
    isin: str | None = None
    ticker: str | None = None
    lei: str | None = None
    other_id: str | None = None
    other_id_desc: str | None = None
    asset_category: str | None = None
    issuer_type: str | None = None
    quantity_units: str | None = None
    currency: str | None = None
    country: str | None = None
    payoff_profile: str | None = None
    fair_value_level: str | None = None
    is_restricted: bool | None = None
    is_cash_collateral: bool | None = None
    is_non_cash_collateral: bool | None = None
    is_loan_by_fund: bool | None = None
    # debt fields
    maturity_date: date | None = None
    coupon_rate: float | None = None
    is_default: bool | None = None
    # derivatives
    derivative_category: str | None = None
    notional_amount: float | None = None


class HoldingsFetcher(Protocol):
    def get_holdings(self, filing: FilingRef, series_id: str | None = None) -> list[Holding]: ...


@dataclass(frozen=True, slots=True)
class PortfolioSnapshot:
    fund: FundClass
    filing: FilingRef
    as_of_date: date | None
    net_assets: float | None
    holdings: list[Holding] = field(default_factory=list)