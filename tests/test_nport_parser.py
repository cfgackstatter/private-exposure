from datetime import date
from private_exposure.sec.nport_source import NportSource

_NS = "http://www.sec.gov/edgar/nport"

def _xml(body: str) -> str:
    return f'<edgarSubmission xmlns="{_NS}"><formData><invstOrSecs>{body}</invstOrSecs></formData></edgarSubmission>'


def _parse(body: str):
    return NportSource._parse(_xml(body), series_id=None)


def test_basic_equity():
    holdings = _parse("""
        <invstOrSec>
            <name>Apple Inc</name>
            <title>APPLE INC COM</title>
            <cusip>037833100</cusip>
            <identifiers><isin value="US0378331005"/><ticker value="AAPL"/></identifiers>
            <valUSD>1000000</valUSD>
            <pctVal>2.5</pctVal>
            <balance>1000</balance>
            <units>NS</units>
            <curCd>USD</curCd>
            <invCountry>US</invCountry>
            <assetCat>EC</assetCat>
            <payoffProfile>long</payoffProfile>
            <fairValLevel>1</fairValLevel>
            <isRestrictedSec>N</isRestrictedSec>
        </invstOrSec>
    """)
    assert len(holdings) == 1
    h = holdings[0]
    assert h.issuer_name == "Apple Inc"
    assert h.title == "APPLE INC COM"
    assert h.cusip == "037833100"
    assert h.isin == "US0378331005"   # attribute, not text
    assert h.ticker == "AAPL"         # attribute, not text
    assert h.value_usd == 1_000_000.0
    assert h.pct_of_net_assets == 2.5
    assert h.is_restricted is False


def test_isin_and_ticker_are_attributes_not_text():
    # regression: isin/ticker use value="" attribute, not element text
    holdings = _parse("""
        <invstOrSec>
            <name>Test Corp</name>
            <identifiers>
                <isin value="US1234567890"/>
                <ticker value="TEST"/>
            </identifiers>
            <valUSD>0</valUSD><pctVal>0</pctVal><balance>0</balance>
        </invstOrSec>
    """)
    assert holdings[0].isin == "US1234567890"
    assert holdings[0].ticker == "TEST"


def test_debt_fields():
    holdings = _parse("""
        <invstOrSec>
            <name>Corp Bond</name>
            <identifiers/>
            <valUSD>500000</valUSD><pctVal>1.0</pctVal><balance>500</balance>
            <debtSec>
                <maturityDt>2030-06-15</maturityDt>
                <annualizedRt>3.25</annualizedRt>
                <isDefault>N</isDefault>
            </debtSec>
        </invstOrSec>
    """)
    h = holdings[0]
    assert h.maturity_date == date(2030, 6, 15)
    assert h.coupon_rate == 3.25
    assert h.is_default is False


def test_missing_optional_fields_are_none():
    holdings = _parse("""
        <invstOrSec>
            <name>Minimal Holding</name>
            <valUSD>100</valUSD><pctVal>0.1</pctVal><balance>1</balance>
        </invstOrSec>
    """)
    h = holdings[0]
    assert h.isin is None
    assert h.ticker is None
    assert h.cusip is None
    assert h.maturity_date is None
    assert h.coupon_rate is None


def test_empty_returns_no_holdings():
    assert NportSource._parse(_xml(""), series_id=None) == []