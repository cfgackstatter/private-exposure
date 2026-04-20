from __future__ import annotations

from datetime import date
from functools import lru_cache

from private_exposure.models import FilingRef
from private_exposure.sec.http import SecHttp


class FilingsSource:
    _BASE = "https://data.sec.gov/submissions/CIK{cik10}.json"
    _PAGE = "https://data.sec.gov/submissions/{filename}"

    def __init__(self, http: SecHttp) -> None:
        self._http = http

    def latest_nport(
        self,
        cik10: str,
        series_id: str | None = None,
        as_of: date | None = None,
    ) -> FilingRef | None:
        candidates = self._nport_candidates(cik10, series_id)
        if as_of is not None:
            candidates = [r for r in candidates if r.report_date <= as_of]  # type: ignore[operator]
        return candidates[0] if candidates else None

    def fetch(
        self,
        cik: int,
        num_quarters: int = 4,
        series_id: str | None = None,
    ) -> list[FilingRef]:
        return self._nport_candidates(str(cik).zfill(10), series_id)[:num_quarters]

    def _nport_candidates(self, cik10: str, series_id: str | None) -> list[FilingRef]:
        """Return N-PORT filings sorted newest-first, optionally filtered by series."""
        candidates = [
            r for r in self._all_filings(cik10)
            if r.form_type in ("NPORT-P", "NPORT-P/A") and r.report_date is not None
        ]
        candidates.sort(key=lambda r: r.report_date, reverse=True)  # type: ignore[arg-type]

        if series_id is None:
            return candidates
        return [r for r in candidates if self._filing_has_series(r, series_id)]

    def _filing_has_series(self, filing: FilingRef, series_id: str) -> bool:
        acc = filing.accession_no.replace("-", "")
        url = (
            f"https://www.sec.gov/Archives/edgar/data/"
            f"{int(filing.cik)}/{acc}/{filing.accession_no}-index.htm"
        )
        text = self._http.get_text(url)

        # if the filing has no series IDs at all, it's a single-series filing — always include it
        import re
        any_series = re.search(r'S\d{9}', text)
        if not any_series:
            return True

        return series_id in text

    @lru_cache(maxsize=256)
    def _all_filings(self, cik10: str) -> list[FilingRef]:
        data = self._http.get_json(self._BASE.format(cik10=cik10))
        rows = self._parse_block(cik10, data.get("filings", {}).get("recent", {}))
        for page in data.get("filings", {}).get("files", []):
            page_data = self._http.get_json(self._PAGE.format(filename=page["name"]))
            rows += self._parse_block(cik10, page_data)
        return rows

    @staticmethod
    def _parse_block(cik10: str, filings: dict) -> list[FilingRef]:
        def _date(s: str) -> date | None:
            try:
                return date.fromisoformat(s) if s else None
            except ValueError:
                return None

        return [
            FilingRef(
                accession_no=acc,
                cik=cik10,
                form_type=ft,
                filing_date=_date(fd) or date.min,
                report_date=_date(rd),
            )
            for ft, acc, fd, rd in zip(
                filings.get("form", []),
                filings.get("accessionNumber", []),
                filings.get("filingDate", []),
                filings.get("reportDate", []),
            )
        ]