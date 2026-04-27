from contextlib import asynccontextmanager
from collections.abc import AsyncGenerator

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from private_exposure.api import admin, search
from private_exposure.db.session import create_db_and_tables
from private_exposure.api.optimize import router as optimize_router

import logging
logging.basicConfig(level=logging.WARNING)
logging.getLogger("private_exposure").setLevel(logging.DEBUG)


@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncGenerator[None, None]:
    create_db_and_tables()
    yield


app = FastAPI(title="Private Exposure", lifespan=lifespan)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:5173", "http://localhost:3000"],
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(admin.router)
app.include_router(search.router)
app.include_router(optimize_router)