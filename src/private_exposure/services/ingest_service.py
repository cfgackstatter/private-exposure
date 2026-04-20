# ingest_service.py
from __future__ import annotations

import re
from dataclasses import asdict
from datetime import date

from sqlmodel import Session, select

from private_exposure.db.models import Filing, Fund, Holding
from private_exposure.sec.filings_source import FilingsSource
from private_exposure.sec.fund_service import FundService
from private_exposure.sec.nport_source import NportSource

_REAL_SERIES_ID = re.compile(r'^S\d{9}$')


class IngestService:
    def __init__(self, funds: FundService, filings: FilingsSource, nport: NportSource, session: Session) -> None:
        self._funds = funds
        self._filings = filings
        self._nport = nport
        self._session = session

    def ingest(self, ticker: str, num_quarters: int = 4, cik: int | None = None, series_id: str | None = None) -> list[date]:
        fund_class = self._funds.find_by_ticker(ticker)
        if fund_class is None and cik is not None:
            fund_class = self._funds.find_by_cik(cik, ticker)
        if fund_class is None:
            raise ValueError(f"Fund not found: {ticker}. Try providing a CIK.")

        fund = self._session.exec(select(Fund).where(Fund.ticker == ticker)).first()
        if fund is None:
            fund = Fund(
                cik=int(fund_class.cik10),
                series_id=series_id or fund_class.series_id,
                class_id=fund_class.class_id,
                ticker=fund_class.ticker,
                company_name=fund_class.company_name,
                series_name=fund_class.series_name,
                class_name=fund_class.class_name,
            )
            self._session.add(fund)
            self._session.flush()

        if fund.id is None:
            raise RuntimeError(f"Failed to persist fund {ticker} — no id after flush")

        # caller-supplied series_id wins; otherwise use stored one only if it's a real SEC ID
        effective_series = series_id or (fund.series_id if _REAL_SERIES_ID.match(fund.series_id) else None)

        nport_filings = self._filings.fetch(fund.cik, series_id=effective_series, num_quarters=num_quarters)
        if not nport_filings:
            raise ValueError(f"No N-PORT filings found for {ticker}.")

        return self._ingest_nport(fund, nport_filings)

    def _ingest_nport(self, fund: Fund, nport_filings: list) -> list[date]:
        assert fund.id is not None
        dates: list[date] = []

        for f in nport_filings:
            already_exists = self._session.exec(
                select(Filing).where(Filing.accession_no == f.accession_no)
            ).first() is not None
            if already_exists:
                continue

            filing = Filing(
                fund_id=fund.id,
                accession_no=f.accession_no,
                form_type=f.form_type,
                filing_date=f.filing_date,
                report_date=f.report_date,
            )
            self._session.add(filing)
            self._session.flush()
            assert filing.id is not None

            self._session.add_all([Holding(filing_id=filing.id, **asdict(h)) for h in self._nport.get_holdings(f)])
            dates.append(f.report_date)

        self._session.commit()
        return dates