from __future__ import annotations

import csv
from functools import cached_property
from io import StringIO

from .fund_catalog import SecFundCatalog
from private_exposure.models import FundClass
from private_exposure.sec.http import SecHttp


class SecFundSource:
    def __init__(self, http: SecHttp, csv_url: str | None = None) -> None:
        self._http = http
        self._csv_url = csv_url
        self._catalog = SecFundCatalog(http)

    @cached_property
    def _fund_classes(self) -> list[FundClass]:
        text = self._http.get_text(self._csv_url or self._catalog.latest_csv_url)
        rows = csv.DictReader(StringIO(text))
        return [self._parse_row(row) for row in rows if (row.get("Class Ticker") or "").strip()]

    @cached_property
    def _ticker_map(self) -> dict[str, FundClass]:
        return {item.ticker: item for item in self._fund_classes}

    def get_fund_classes(self) -> list[FundClass]:
        return self._fund_classes

    def find_by_ticker(self, ticker: str) -> FundClass | None:
        return self._ticker_map.get(ticker.strip().upper())

    def find_by_cik(self, cik: int, ticker: str) -> FundClass | None:
        data = self._http.get_json(
            f"https://data.sec.gov/submissions/CIK{str(cik).zfill(10)}.json"
        )
        name = data.get("name", "")
        if not name:
            return None
        return FundClass(
            cik=cik,
            series_id=f"CIK{cik}",   # synthetic — interval funds have no series ID
            class_id=f"CIK{cik}",
            ticker=ticker.upper(),
            company_name=name,
            series_name=name,
            class_name=name,
        )

    @staticmethod
    def _parse_row(row: dict[str, str | None]) -> FundClass:
        r = {k.strip(): (v or "").strip() for k, v in row.items()}
        return FundClass(
            cik=int(r.get("CIK Number") or "0"),
            series_id=r.get("Series ID", ""),
            class_id=r.get("Class ID", ""),
            ticker=r.get("Class Ticker", "").upper(),
            company_name=r.get("Entity Name", ""),
            series_name=r.get("Series Name", ""),
            class_name=r.get("Class Name", ""),
        )