# api/search.py
from __future__ import annotations

import re
from dataclasses import dataclass

from fastapi import APIRouter, Depends
from sqlmodel import Session, select

from private_exposure.api.deps import get_session
from private_exposure.db.models import Filing, Fund, Holding

router = APIRouter(prefix="/search", tags=["search"])


# --- query parsing ---

@dataclass
class Token:
    keywords: list[str]  # all must match (AND group)

def parse_query(q: str) -> list[Token]:
    """Parse 'space AND launch OR uranium' into OR-of-AND tokens."""
    or_parts = re.split(r"\bOR\b", q, flags=re.IGNORECASE)
    return [
        Token(keywords=[kw.strip().lower() for kw in re.split(r"\bAND\b", part, flags=re.IGNORECASE) if kw.strip()])
        for part in or_parts
        if part.strip()
    ]

def matches(text: str, tokens: list[Token]) -> bool:
    """True if text satisfies any OR token (each token requires all AND keywords)."""
    lower = text.lower()
    return any(all(kw in lower for kw in token.keywords) for token in tokens)


# --- response models ---

from pydantic import BaseModel

class MatchedHolding(BaseModel):
    issuer_name: str
    title: str | None
    ticker: str | None
    isin: str | None
    cusip: str | None
    asset_category: str | None
    country: str | None
    pct_of_net_assets: float

class FundSearchResult(BaseModel):
    id: int
    ticker: str
    series_name: str
    class_name: str
    cik: int
    total_weight: float
    match_count: int
    matched_holdings: list[MatchedHolding]
    report_date: str


# --- endpoint ---

@router.get("", response_model=list[FundSearchResult])
def search(q: str, session: Session = Depends(get_session)) -> list[FundSearchResult]:
    if not q.strip():
        return []

    tokens = parse_query(q)
    funds = session.exec(select(Fund)).all()
    results: list[FundSearchResult] = []

    for fund in funds:
        # latest filing only
        filing = session.exec(
            select(Filing)
            .where(Filing.fund_id == fund.id)
            .order_by(Filing.report_date.desc())  # type: ignore[attr-defined]
        ).first()
        if filing is None:
            continue

        holdings = session.exec(
            select(Holding).where(Holding.filing_id == filing.id)
        ).all()

        matched = [h for h in holdings if matches(h.issuer_name, tokens)]
        if not matched:
            continue

        matched.sort(key=lambda h: h.pct_of_net_assets, reverse=True)
        results.append(FundSearchResult(
            id=fund.id,  # type: ignore[arg-type]
            ticker=fund.ticker,
            series_name=fund.series_name,
            class_name=fund.class_name,
            cik=fund.cik,
            total_weight=round(sum(h.pct_of_net_assets for h in matched), 4),
            match_count=len(matched),
            matched_holdings=[
                MatchedHolding(
                    issuer_name=h.issuer_name,
                    title=h.title,
                    ticker=h.ticker,
                    isin=h.isin,
                    cusip=h.cusip,
                    asset_category=h.asset_category,
                    country=h.country,
                    pct_of_net_assets=h.pct_of_net_assets,
                )
                for h in matched  # no [:5] slice
            ],
            report_date=str(filing.report_date),
        ))

    results.sort(key=lambda r: r.total_weight, reverse=True)
    return results