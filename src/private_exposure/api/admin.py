# admin.py
from __future__ import annotations

from fastapi import APIRouter, Depends, HTTPException
from pydantic import BaseModel
from sqlmodel import Session, select

from private_exposure.api.deps import get_ingest_service, get_session
from private_exposure.schemas.holdings import FundOut, FilingOut, HoldingOut, IngestResult
from private_exposure.db.models import Filing, Fund, Holding

router = APIRouter(prefix="/admin", tags=["admin"])


class IngestRequest(BaseModel):
    cik: int | None = None
    series_id: str | None = None
    num_quarters: int = 4


def _fund_out(f: Fund, session: Session) -> FundOut:
    filings = session.exec(select(Filing).where(Filing.fund_id == f.id)).all()
    return FundOut(
        id=f.id,  # type: ignore[arg-type]
        cik=f.cik,
        ticker=f.ticker,
        series_id=f.series_id,
        company_name=f.company_name,
        series_name=f.series_name,
        class_name=f.class_name,
        filings=[
            FilingOut(
                id=fi.id,  # type: ignore[arg-type]
                accession_no=fi.accession_no,
                form_type=fi.form_type,
                filing_date=fi.filing_date,
                report_date=fi.report_date,
            )
            for fi in filings
        ],
    )


@router.post("/funds/{ticker}", response_model=IngestResult)
def ingest_fund(
    ticker: str,
    body: IngestRequest = IngestRequest(),
    session: Session = Depends(get_session),
) -> IngestResult:
    try:
        dates = get_ingest_service(session).ingest(
            ticker.upper(),
            num_quarters=body.num_quarters,
            cik=body.cik,
            series_id=body.series_id,
        )
    except ValueError as e:
        raise HTTPException(status_code=404, detail=str(e))
    return IngestResult(ticker=ticker.upper(), dates_ingested=dates)


@router.get("/funds", response_model=list[FundOut])
def list_funds(session: Session = Depends(get_session)) -> list[FundOut]:
    return [_fund_out(f, session) for f in session.exec(select(Fund)).all()]


@router.get("/funds/search", response_model=FundOut | None)
def search_fund(ticker: str, session: Session = Depends(get_session)) -> FundOut | None:
    fund = session.exec(select(Fund).where(Fund.ticker == ticker.upper())).first()
    return _fund_out(fund, session) if fund else None


@router.delete("/funds/{ticker}", status_code=204)
def delete_fund(ticker: str, session: Session = Depends(get_session)) -> None:
    fund = session.exec(select(Fund).where(Fund.ticker == ticker.upper())).first()
    if fund is None:
        raise HTTPException(status_code=404, detail=f"Fund {ticker.upper()} not found")
    session.delete(fund)
    session.commit()


@router.get("/funds/{ticker}/filings/{filing_id}/holdings", response_model=list[HoldingOut])
def get_holdings(ticker: str, filing_id: int, session: Session = Depends(get_session)) -> list[HoldingOut]:
    holdings = session.exec(select(Holding).where(Holding.filing_id == filing_id)).all()
    if not holdings:
        raise HTTPException(status_code=404, detail="No holdings found")
    return [HoldingOut.model_validate(h, from_attributes=True) for h in holdings]