from collections.abc import Generator

import os
from dotenv import load_dotenv
load_dotenv()

from sqlmodel import Session, SQLModel, create_engine

from private_exposure.db import models as _models  # noqa: F401 — registers tables

_DATABASE_URL = os.environ.get(
    "DATABASE_URL",
    "postgresql://postgres:postgres@localhost:5433/private_exposure",
)

engine = create_engine(_DATABASE_URL, echo=False)


def create_db_and_tables() -> None:
    SQLModel.metadata.create_all(engine)


def get_session() -> Generator[Session, None, None]:
    with Session(engine) as session:
        yield session