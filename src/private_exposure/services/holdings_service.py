from __future__ import annotations

from datetime import date

from private_exposure.models import FilingLookup, FundLookup, HoldingsFetcher, PortfolioSnapshot


class HoldingsService:
    def __init__(
        self,
        funds: FundLookup,
        filings: FilingLookup,
        nport: HoldingsFetcher,
    ) -> None:
        self._funds = funds
        self._filings = filings
        self._nport = nport

    def get_portfolio(
        self,
        ticker: str,
        as_of: date | None = None,
    ) -> PortfolioSnapshot | None:
        fund = self._funds.find_by_ticker(ticker)
        if fund is None:
            return None

        filing = self._filings.latest_nport(fund.cik10, series_id=fund.series_id, as_of=as_of)
        if filing is None:
            return None

        holdings = self._nport.get_holdings(filing, series_id=fund.series_id)
        return PortfolioSnapshot(
            fund=fund,
            filing=filing,
            as_of_date=filing.report_date,
            net_assets=None,
            holdings=holdings,
        )