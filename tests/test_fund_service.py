import pytest
from private_exposure.models import FundClass
from private_exposure.sec.fund_catalog import SecFundCatalog
from private_exposure.sec.fund_service import FundService
from private_exposure.sec.fund_source import SecFundSource
from private_exposure.sec.http import SecHttp


class FakeFundSource:
    _items = [
        FundClass(cik=1000001, series_id="S000000001", class_id="C000000001",
                  ticker="AAAAX", company_name="Alpha Funds",
                  series_name="Alpha Growth Fund", class_name="Investor Class"),
        FundClass(cik=1000002, series_id="S000000002", class_id="C000000002",
                  ticker="BBBIX", company_name="Beta Funds",
                  series_name="Beta Value Fund", class_name="Institutional Class"),
    ]
    _map = {item.ticker: item for item in _items}

    def get_fund_classes(self): return self._items
    def find_by_ticker(self, ticker): return self._map.get(ticker.strip().upper())
    def find_by_cik(self, cik, ticker): return None


def test_find_cik_by_ticker():
    svc = FundService(FakeFundSource())
    assert svc.find_cik_by_ticker("aaaax") == "0001000001"
    assert svc.find_cik_by_ticker("BBBIX") == "0001000002"
    assert svc.find_cik_by_ticker("missing") is None


def test_find_by_ticker_case_insensitive():
    svc = FundService(FakeFundSource())
    assert svc.find_by_ticker("aaaax") == svc.find_by_ticker("AAAAX")


@pytest.mark.integration
def test_csv_url_from_sec():
    http = SecHttp("private-exposure your-email@example.com")
    try:
        url = SecFundCatalog(http).latest_csv_url
        assert url.startswith("https://www.sec.gov/files/")
        assert url.endswith(".csv")
    finally:
        http.close()


@pytest.mark.integration
def test_find_real_fund_from_sec():
    http = SecHttp("private-exposure your-email@example.com")
    try:
        item = FundService(SecFundSource(http)).find_by_ticker("VTSAX")
        assert item is not None
        assert item.ticker == "VTSAX"
        assert len(item.cik10) == 10
        assert item.series_id.startswith("S")
        assert item.class_id.startswith("C")
    finally:
        http.close()