from datetime import date
from typing import List, Optional

from sqlmodel import Field, Relationship, SQLModel


class Fund(SQLModel, table=True):
    __tablename__ = "funds"  # type: ignore[assignment]

    id: Optional[int] = Field(default=None, primary_key=True)
    cik: int = Field(index=True)
    series_id: str = Field(index=True, unique=True)
    class_id: str
    ticker: str = Field(index=True, unique=True)
    company_name: str
    series_name: str
    class_name: str

    filings: list["Filing"] = Relationship(
        back_populates="fund",
        sa_relationship_kwargs={"cascade": "all, delete-orphan"},
    )


class Filing(SQLModel, table=True):
    __tablename__ = "filings"  # type: ignore[assignment]

    id: Optional[int] = Field(default=None, primary_key=True)
    fund_id: int = Field(foreign_key="funds.id", index=True)
    accession_no: str = Field(index=True, unique=True)
    form_type: str
    filing_date: date
    report_date: date = Field(index=True)

    fund: Optional["Fund"] = Relationship(back_populates="filings")
    holdings: list["Holding"] = Relationship(
        back_populates="filing",
        sa_relationship_kwargs={"cascade": "all, delete-orphan"},
    )


class Holding(SQLModel, table=True):
    __tablename__ = "holdings"  # type: ignore[assignment]

    id: Optional[int] = Field(default=None, primary_key=True)
    filing_id: int = Field(foreign_key="filings.id", index=True)

    issuer_name: str
    title: Optional[str] = None
    cusip: Optional[str] = None
    isin: Optional[str] = None
    ticker: Optional[str] = Field(default=None, index=True)
    lei: Optional[str] = None
    other_id: Optional[str] = None
    other_id_desc: Optional[str] = None
    asset_category: Optional[str] = None
    issuer_type: Optional[str] = None
    value_usd: float
    pct_of_net_assets: float
    quantity: float
    quantity_units: Optional[str] = None
    currency: Optional[str] = None
    country: Optional[str] = None
    payoff_profile: Optional[str] = None
    fair_value_level: Optional[str] = None
    is_restricted: Optional[bool] = None
    is_cash_collateral: Optional[bool] = None
    is_non_cash_collateral: Optional[bool] = None
    is_loan_by_fund: Optional[bool] = None
    # debt
    maturity_date: Optional[date] = Field(default=None)
    coupon_rate: Optional[float] = Field(default=None)
    is_default: Optional[bool] = Field(default=None)
    # derivatives
    derivative_category: Optional[str] = None
    notional_amount: Optional[float] = None

    filing: Optional["Filing"] = Relationship(back_populates="holdings")