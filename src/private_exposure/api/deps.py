from collections.abc import Generator
from functools import lru_cache

from sqlmodel import Session

from private_exposure.db.session import engine
from private_exposure.sec.http import SecHttp
from private_exposure.sec.fund_source import SecFundSource
from private_exposure.sec.fund_service import FundService
from private_exposure.sec.filings_source import FilingsSource
from private_exposure.sec.nport_source import NportSource
from private_exposure.services.ingest_service import IngestService


@lru_cache
def _http() -> SecHttp:
    return SecHttp("private-exposure your-email@example.com")


@lru_cache
def get_ingest_service_deps() -> tuple[FundService, FilingsSource, NportSource]:
    http = _http()
    return (
        FundService(SecFundSource(http)),
        FilingsSource(http),
        NportSource(http),
    )


def get_session() -> Generator[Session, None, None]:
    with Session(engine) as session:
        yield session


def get_ingest_service(session: Session) -> IngestService:
    funds, filings, nport = get_ingest_service_deps()
    return IngestService(
        funds=funds,
        filings=filings,
        nport=nport,
        session=session,
    )