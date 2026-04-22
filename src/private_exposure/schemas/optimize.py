from __future__ import annotations
from pydantic import BaseModel, Field


class OptimizeRequest(BaseModel):
    keywords: list[str]
    budget: float = 100_000.0
    max_gross_leverage: float = 2.0
    max_fund_weight: float = 0.35
    max_short_weight: float = 0.10
    borrow_cost: float = 0.05
    financing_cost: float = 0.03
    holding_period_months: int = 12
    max_funds: int = 10
    max_hedges: int = 30


class PositionOut(BaseModel):
    name: str
    identifier: str | None
    weight: float
    dollar_amount: float
    annual_cost_pct: float = 0.0


class PortfolioMetricsOut(BaseModel):
    target_exposure_pct: float
    hedgeable_residual_pct: float
    unhedgeable_residual_pct: float
    gross_leverage: float
    net_exposure: float
    annual_carry_cost_pct: float
    long_count: int
    short_count: int


class PortfolioOut(BaseModel):
    label: str
    metrics: PortfolioMetricsOut
    longs: list[PositionOut]
    shorts: list[PositionOut]
    unhedgeable_names: list[str]


class OptimizeResponse(BaseModel):
    keywords: list[str]
    portfolios: list[PortfolioOut]
    warnings: list[str] = Field(default_factory=list)