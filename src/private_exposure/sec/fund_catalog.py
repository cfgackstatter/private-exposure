from __future__ import annotations

import re
from functools import cached_property

from private_exposure.sec.http import SecHttp


class SecFundCatalog:
    _PAGE_URL = (
        "https://www.sec.gov/data-research/sec-markets-data/"
        "investment-company-series-class-information"
    )
    _CSV_RE = re.compile(
        r"/files/investment/data/other/"
        r"investment-company-series-class-information/"
        r"investment-company-series-class-(\d{4})\.csv"
    )

    def __init__(self, http: SecHttp) -> None:
        self._http = http

    @cached_property
    def latest_csv_url(self) -> str:
        text = self._http.get_text(self._PAGE_URL)
        years = [int(value) for value in self._CSV_RE.findall(text)]
        if not years:
            raise ValueError("Could not find SEC fund CSV links.")
        year = max(years)
        return (
            "https://www.sec.gov/files/investment/data/other/"
            "investment-company-series-class-information/"
            f"investment-company-series-class-{year}.csv"
        )