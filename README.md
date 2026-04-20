# Private Exposure

Thematic fund holdings screener. Search across SEC N-PORT filings to find
and analyze fund exposure to any company, sector, or topic.

## Stack

- **Backend** — FastAPI + SQLModel + PostgreSQL
- **Frontend** — React + Vite + TypeScript

## Setup

### 1. Database
```bash
sudo service postgresql start
```

### 2. Backend
```bash
pip install -e .
uvicorn private_exposure.main:app --reload
```

### 3. Frontend
```bash
cd frontend && npm install && npm run dev
```

Copy `.env.example` to `.env` and set `DATABASE_URL` if your Postgres config differs from the default.

## Routes

| URL | Description |
|---|---|
| `http://localhost:5173/` | User search page |
| `http://localhost:5173/admin` | Admin — ingest and manage funds |
| `http://localhost:8000/docs` | FastAPI auto-docs |

## Tests

```bash
pytest                        # unit tests only
pytest -m integration         # also hits live SEC EDGAR
```