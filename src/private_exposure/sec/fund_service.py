from private_exposure.models import FundClass, FundClassReader


class FundService:
    def __init__(self, source: FundClassReader) -> None:
        self._source = source

    def find_by_ticker(self, ticker: str) -> FundClass | None:
        return self._source.find_by_ticker(ticker)

    def find_cik_by_ticker(self, ticker: str) -> str | None:
        item = self.find_by_ticker(ticker)
        return None if item is None else item.cik10

    def find_by_cik(self, cik: int, ticker: str) -> FundClass | None:
        return self._source.find_by_cik(cik, ticker)