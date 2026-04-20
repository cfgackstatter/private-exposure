from __future__ import annotations
import re
from datetime import date
from unittest.mock import MagicMock

import pytest

from private_exposure.models import FundClass, FilingRef
from private_exposure.services.ingest_service import IngestService, _REAL_SERIES_ID


def _filing(accession="0000001-24-000001", report=date(2024, 1, 31)):
    return FilingRef(accession_no=accession, cik="0000000001",
                     form_type="NPORT-P", filing_date=date(2024, 3, 1),
                     report_date=report)


def _fund_class(series_id="S000000001"):
    return FundClass(cik=1, series_id=series_id, class_id="C000000001",
                     ticker="FAKE", company_name="Fake Funds",
                     series_name="Fake Fund", class_name="Investor")


def test_real_series_id_regex():
    assert _REAL_SERIES_ID.match("S000096481")
    assert not _REAL_SERIES_ID.match("CIK2081199")
    assert not _REAL_SERIES_ID.match("S12345")  # too short


def test_synthetic_series_id_skips_filter(tmp_path):
    """When series_id is synthetic (CIKxxxxx), fetch is called with series_id=None."""
    funds = MagicMock()
    funds.find_by_ticker.return_value = _fund_class(series_id="CIK2081199")

    filings = MagicMock()
    filings.fetch.return_value = []

    svc = IngestService(funds, filings, MagicMock(), MagicMock())
    with pytest.raises(ValueError, match="No N-PORT"):
        svc.ingest("FAKE")

    _, kwargs = filings.fetch.call_args
    assert kwargs.get("series_id") is None


def test_real_series_id_passes_through():
    """When series_id is real (S000xxxxxx), it's passed to fetch."""
    funds = MagicMock()
    funds.find_by_ticker.return_value = _fund_class(series_id="S000096481")

    filings = MagicMock()
    filings.fetch.return_value = []

    svc = IngestService(funds, filings, MagicMock(), MagicMock())
    with pytest.raises(ValueError, match="No N-PORT"):
        svc.ingest("FAKE")

    _, kwargs = filings.fetch.call_args
    assert kwargs.get("series_id") == "S000096481"


def test_caller_series_id_overrides_stored():
    """Explicit series_id param always wins over stored value."""
    funds = MagicMock()
    funds.find_by_ticker.return_value = _fund_class(series_id="CIK2081199")

    filings = MagicMock()
    filings.fetch.return_value = []

    svc = IngestService(funds, filings, MagicMock(), MagicMock())
    with pytest.raises(ValueError, match="No N-PORT"):
        svc.ingest("FAKE", series_id="S000096481")

    _, kwargs = filings.fetch.call_args
    assert kwargs.get("series_id") == "S000096481"