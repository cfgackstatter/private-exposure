from __future__ import annotations
from pydantic import BaseModel, Field


class OptimizeRequest(BaseModel):
    keywords: list[str]
    budget: float = 100_000
    max_gross_leverage: float = 3.0
    max_fund_weight: float | None = None
    borrow_cost: float = 0.05
    financing_cost: float = 0.03
    holding_period_months: int = 12
    max_funds: int = 10
    max_hedges: int = 30
    short_cap: float = 0.5
    

class PositionOut(BaseModel):
    name: str
    identifier: str
    weight: float
    dollar_amount: float
    annual_cost_pct: float
    alpha_pct: float | None = None   # IC-scaled expected return, annualised %


class PortfolioMetricsOut(BaseModel):
    target_exposure_pct: float      # NAV% in theme — what you want
    non_target_long_pct: float      # NAV% in everything else you're long
    short_total_pct: float          # NAV% short (total short book)
    net_non_target_pct: float       # non_target_long - short_total (residual after hedging)
    unhedgeable_pct: float          # subset of non-target that cannot be shorted
    gross_leverage: float           # w.sum() + s.sum()
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