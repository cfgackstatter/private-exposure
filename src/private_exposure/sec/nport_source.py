from __future__ import annotations

import re
import xml.etree.ElementTree as ET
from datetime import date
from functools import lru_cache

from private_exposure.models import FilingRef, Holding
from private_exposure.sec.http import SecHttp

_NS = "http://www.sec.gov/edgar/nport"


def _txt(el: ET.Element | None) -> str | None:
    return el.text.strip() if el is not None and el.text else None


def _float(el: ET.Element | None) -> float | None:
    t = _txt(el)
    try:
        return float(t) if t else None
    except ValueError:
        return None


def _bool(el: ET.Element | None) -> bool | None:
    t = _txt(el)
    return None if t is None else t.upper() in ("Y", "YES", "TRUE", "1")


def _date(el: ET.Element | None) -> date | None:
    t = _txt(el)
    if not t:
        return None
    try:
        return date.fromisoformat(t)
    except ValueError:
        return None


class NportSource:
    def __init__(self, http: SecHttp) -> None:
        self._http = http

    def get_holdings(self, filing: FilingRef, series_id: str | None = None) -> list[Holding]:
        xml_text = self._fetch_xml(filing)
        return self._parse(xml_text, series_id)

    @lru_cache(maxsize=256)
    def _fetch_xml(self, filing: FilingRef) -> str:
        acc_clean = filing.accession_no.replace("-", "")
        index_url = (
            f"https://www.sec.gov/Archives/edgar/data/"
            f"{int(filing.cik)}/{acc_clean}/{filing.accession_no}-index.htm"
        )
        index_html = self._http.get_text(index_url)
        filename = self._extract_primary_xml(index_html)
        return self._http.get_text(
            f"https://www.sec.gov/Archives/edgar/data/"
            f"{int(filing.cik)}/{acc_clean}/{filename}"
        )

    @staticmethod
    def _extract_primary_xml(index_html: str) -> str:
        matches = re.findall(r'href="([^"]+\.xml)"', index_html, re.IGNORECASE)
        if not matches:
            raise ValueError("Could not find XML document in N-PORT filing index.")
        return matches[0].split("/")[-1]

    @staticmethod
    def _parse(xml_text: str, series_id: str | None) -> list[Holding]:
        root = ET.fromstring(xml_text)
        ns = {"n": _NS} if _NS in xml_text else {}

        def find(el: ET.Element, path: str) -> ET.Element | None:
            return el.find(path, ns) if ns else el.find(path.split(":")[-1] if ":" in path else path)

        if series_id:
            search_root = None
            for el in root.iter():
                if not el.tag.endswith("seriesInfo"):
                    continue
                for child in el:
                    if child.get("seriesId") == series_id:
                        search_root = child
                        break
                if search_root is not None:
                    break
            search_root = search_root or root
        else:
            search_root = root

        holdings = []
        for inv in search_root.iter():
            if not inv.tag.endswith("invstOrSec"):
                continue

            isin_el = find(inv, "n:identifiers/n:isin")
            ticker_el = find(inv, "n:identifiers/n:ticker")
            other_el = find(inv, "n:identifiers/n:other")
            sec_lending = find(inv, "n:securityLending")
            debt = find(inv, "n:debtSec")

            holdings.append(Holding(
                issuer_name=_txt(find(inv, "n:name")) or "",
                title=_txt(find(inv, "n:title")),
                cusip=_txt(find(inv, "n:cusip")),
                isin=isin_el.get("value") if isin_el is not None else None,
                ticker=ticker_el.get("value") if ticker_el is not None else None,
                lei=_txt(find(inv, "n:lei")),
                other_id=other_el.get("value") if other_el is not None else None,
                other_id_desc=other_el.get("otherDesc") if other_el is not None else None,
                asset_category=_txt(find(inv, "n:assetCat")),
                issuer_type=_txt(find(inv, "n:issuerCat")),
                value_usd=_float(find(inv, "n:valUSD")) or 0.0,
                pct_of_net_assets=_float(find(inv, "n:pctVal")) or 0.0,
                quantity=_float(find(inv, "n:balance")) or 0.0,
                quantity_units=_txt(find(inv, "n:units")),
                currency=_txt(find(inv, "n:curCd")),
                country=_txt(find(inv, "n:invCountry")),
                payoff_profile=_txt(find(inv, "n:payoffProfile")),
                fair_value_level=_txt(find(inv, "n:fairValLevel")),
                is_restricted=_bool(find(inv, "n:isRestrictedSec")),
                is_cash_collateral=_bool(find(sec_lending, "n:isCashCollateral")) if sec_lending is not None else None,
                is_non_cash_collateral=_bool(find(sec_lending, "n:isNonCashCollateral")) if sec_lending is not None else None,
                is_loan_by_fund=_bool(find(sec_lending, "n:isLoanByFund")) if sec_lending is not None else None,
                maturity_date=_date(find(debt, "n:maturityDt")) if debt is not None else None,
                coupon_rate=_float(find(debt, "n:annualizedRt")) if debt is not None else None,
                is_default=_bool(find(debt, "n:isDefault")) if debt is not None else None,
                derivative_category=_txt(find(inv, "n:derivCat")),
                notional_amount=_float(find(inv, "n:notionalAmt")),
            ))
        return holdings