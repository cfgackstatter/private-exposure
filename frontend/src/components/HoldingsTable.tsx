import { useEffect, useState } from "react";
import { api } from "../api/client";
import type { HoldingOut } from "../types";
import { identifier } from "../utils/holding";

interface Props {
  ticker: string;
  filingId: number;
  reportDate: string;
}

export function HoldingsTable({ ticker, filingId, reportDate }: Props) {
  const [holdings, setHoldings] = useState<HoldingOut[]>([]);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState<string | null>(null);

  useEffect(() => {
    setLoading(true);
    setError(null);
    api.getHoldings(ticker, filingId)
      .then(setHoldings)
      .catch(e => setError(e.message))
      .finally(() => setLoading(false));
  }, [ticker, filingId]);

  if (loading) return <div className="loading-row">Loading holdings…</div>;
  if (error) return <div className="error-row">Error: {error}</div>;

  return (
    <div className="holdings-section">
      <h3>{ticker} — Holdings as of {reportDate} ({holdings.length})</h3>
      <div className="table-wrap table-wrap--scrollable">
        <table>
          <thead>
            <tr>
              <th>%</th>
              <th>Name</th>
              <th>Identifier</th>
              <th>Category</th>
              <th>Country</th>
              <th className="num">Value (USD)</th>
            </tr>
          </thead>
          <tbody>
            {holdings
              .sort((a, b) => b.pct_of_net_assets - a.pct_of_net_assets)
              .map((h, i) => (
                <tr key={i}>
                  <td className="num">{h.pct_of_net_assets.toFixed(2)}%</td>
                  <td className="col-issuer" title={h.issuer_name}>{h.title ?? h.issuer_name}</td>
                  <td>{identifier(h)}</td>
                  <td>{h.asset_category ?? "—"}</td>
                  <td>{h.country ?? "—"}</td>
                  <td className="num">{h.value_usd.toLocaleString("en-US", { style: "currency", currency: "USD", maximumFractionDigits: 0 })}</td>
                </tr>
              ))}
          </tbody>
        </table>
      </div>
    </div>
  );
}