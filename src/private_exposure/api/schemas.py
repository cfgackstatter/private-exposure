from datetime import date
from typing import Optional
from pydantic import BaseModel


class HoldingOut(BaseModel):
    issuer_name: str
    title: Optional[str]
    cusip: Optional[str]
    isin: Optional[str]
    ticker: Optional[str]
    lei: Optional[str]
    other_id: Optional[str]
    other_id_desc: Optional[str]
    asset_category: Optional[str]
    issuer_type: Optional[str]
    value_usd: float
    pct_of_net_assets: float
    quantity: float
    quantity_units: Optional[str]
    currency: Optional[str]
    country: Optional[str]
    payoff_profile: Optional[str]
    fair_value_level: Optional[str]
    is_restricted: Optional[bool]
    is_cash_collateral: Optional[bool]
    is_non_cash_collateral: Optional[bool]
    is_loan_by_fund: Optional[bool]
    maturity_date: Optional[date]
    coupon_rate: Optional[float]
    is_default: Optional[bool]
    derivative_category: Optional[str]
    notional_amount: Optional[float]


class FilingOut(BaseModel):
    id: int
    accession_no: str
    form_type: str
    filing_date: date
    report_date: date
    holdings: list[HoldingOut] = []


class FundOut(BaseModel):
    id: int
    cik: int
    ticker: str
    series_id: str
    company_name: str
    series_name: str
    class_name: str
    filings: list[FilingOut] = []


class IngestResult(BaseModel):
    ticker: str
    dates_ingested: list[date]