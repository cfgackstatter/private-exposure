import { useState } from "react";
import { api } from "../api/client";

interface Props {
  onIngested: () => void;
}

export function IngestPanel({ onIngested }: Props) {
  const [ticker, setTicker] = useState("");
  const [cik, setCik] = useState("");
  const [seriesId, setSeriesId] = useState("");
  const [quarters, setQuarters] = useState(4);
  const [status, setStatus] = useState<string | null>(null);
  const [error, setError] = useState<string | null>(null);
  const [loading, setLoading] = useState(false);

  async function handleLoad() {
    if (!ticker.trim()) return;
    setLoading(true);
    setStatus(null);
    setError(null);
    try {
      const result = await api.ingestFund(
        ticker.trim().toUpperCase(),
        quarters,
        cik ? parseInt(cik) : undefined,
        seriesId.trim() || undefined,
      );
      if (result.dates_ingested.length === 0) {
        setStatus(`${result.ticker}: already up to date`);
      } else {
        setStatus(`${result.ticker}: loaded ${result.dates_ingested.length} filing(s) — ${result.dates_ingested.join(", ")}`);
      }
      onIngested();
    } catch (e: unknown) {
      setError(e instanceof Error ? e.message : "Unknown error");
    } finally {
      setLoading(false);
    }
  }

  return (
    <div className="ingest-panel">
      <h2>Load Fund Data</h2>
      <div className="ingest-row">
        <input
          className="input"
          placeholder="Ticker (e.g. BFGFX)"
          value={ticker}
          onChange={e => setTicker(e.target.value.toUpperCase())}
          onKeyDown={e => e.key === "Enter" && handleLoad()}
          disabled={loading}
        />
        <input
          className="input"
          placeholder="CIK (optional)"
          value={cik}
          onChange={e => setCik(e.target.value.replace(/\D/g, ""))}
          disabled={loading}
          style={{ width: "120px" }}
        />
        {cik && (
          <input
            className="input"
            placeholder="Series ID (e.g. S000096481)"
            value={seriesId}
            onChange={e => setSeriesId(e.target.value.trim())}
            disabled={loading}
            style={{ width: "180px" }}
          />
        )}
        <select
          className="input select"
          value={quarters}
          onChange={e => setQuarters(Number(e.target.value))}
          disabled={loading}
        >
          {[1, 2, 4, 8, 12].map(q => (
            <option key={q} value={q}>{q} quarter{q > 1 ? "s" : ""}</option>
          ))}
        </select>
        <button className="btn btn-primary" onClick={handleLoad} disabled={loading}>
          {loading ? "Loading…" : "Load"}
        </button>
      </div>
      {status && <p className="status-msg status-ok">✓ {status}</p>}
      {error && <p className="status-msg status-err">✗ {error}</p>}
    </div>
  );
}