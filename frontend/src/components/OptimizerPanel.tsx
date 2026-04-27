import { useState } from "react";
import { api } from "../api/client";
import type { OptimizeResponse, PortfolioOut, OptimizeRequest, PositionOut } from "../types";

interface Props {
  keywords: string[];
}

const DEFAULTS: Omit<OptimizeRequest, "keywords"> = {
  budget: 100_000,
  max_gross_leverage: 3.0,
  max_fund_weight: null,
  borrow_cost: 0.05,
  financing_cost: 0.03,
  holding_period_months: 12,
  max_funds: 10,
  max_hedges: 30,
  short_cap: 0.5,
};

export function OptimizerPanel({ keywords }: Props) {
  const [showAdvanced, setShowAdvanced] = useState(false);
  const [params, setParams] = useState(DEFAULTS);
  const [result, setResult] = useState<OptimizeResponse | null>(null);
  const [selected, setSelected] = useState<string>("Efficient");
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);

  function setParam<K extends keyof typeof DEFAULTS>(k: K, v: (typeof DEFAULTS)[K]) {
    setParams(p => ({ ...p, [k]: v }));
  }

  async function run() {
    setLoading(true);
    setError(null);
    setResult(null);
    try {
      const res = await api.optimize({ keywords, ...params });
      setResult(res);
      const labels = res.portfolios.map((p: PortfolioOut) => p.label);
      setSelected(labels.includes("Efficient") ? "Efficient" : labels[0] ?? "");
    } catch (e: any) {
      setError(e.message);
    } finally {
      setLoading(false);
    }
  }

  const portfolio = result?.portfolios.find(p => p.label === selected) ?? null;

  return (
    <div className="optimizer-panel">
      <div className="optimizer-header">
        <h3>Optimize Exposure</h3>
        <div className="optimizer-controls">
          <label>
            Budget ($)
            <input
              type="number"
              value={params.budget}
              onChange={e => setParam("budget", Number(e.target.value))}
            />
          </label>
          <label>
            Max Gross Leverage
            <input
              type="number"
              step="0.5"
              value={params.max_gross_leverage}
              onChange={e => setParam("max_gross_leverage", Number(e.target.value))}
            />
          </label>
          <button
            className="btn-link"
            onClick={() => setShowAdvanced(v => !v)}
          >
            {showAdvanced ? "Hide advanced" : "Advanced"}
          </button>
        </div>

        {showAdvanced && (
          <div className="optimizer-advanced">
            <label>
              Borrow Cost (annual)
              <input type="number" step="0.01"
                value={params.borrow_cost}
                onChange={e => setParam("borrow_cost", Number(e.target.value))} />
            </label>
            <label>
              Financing Cost (annual)
              <input type="number" step="0.01"
                value={params.financing_cost}
                onChange={e => setParam("financing_cost", Number(e.target.value))} />
            </label>
            <label>
              Holding Period (months)
              <input type="number" step="1"
                value={params.holding_period_months}
                onChange={e => setParam("holding_period_months", Number(e.target.value))} />
            </label>
            <label>
              Short Cap
              <input type="number" step="0.05" min="0.05" max="1"
                value={params.short_cap}
                onChange={e => setParam("short_cap", Number(e.target.value))} />
            </label>
            <label>
              Max Hedges
              <input type="number" min={5} max={100} step={5}
                value={params.max_hedges}
                onChange={e => setParam("max_hedges", Number(e.target.value))}
              />
            </label>
        </div>
        )}

        <button className="btn btn-primary" onClick={run} disabled={loading}>
          {loading ? "Optimizing…" : "Generate portfolios"}
        </button>
      </div>

      {error && <div className="error-row">Error: {error}</div>}

      {result && (
        <>
          {result.warnings.length > 0 && (
            <div className="optimizer-warnings">
              {result.warnings.map((w, i) => <p key={i}>⚠ {w}</p>)}
            </div>
          )}

          {/* frontier tabs */}
          <div className="frontier-tabs">
            {result.portfolios.map(p => (
              <button
                key={p.label}
                className={`frontier-tab ${p.label === selected ? "active" : ""}`}
                onClick={() => setSelected(p.label)}
              >
                {p.label}
                <span className="tab-target">{p.metrics.target_exposure_pct.toFixed(1)}% target</span>
              </button>
            ))}
          </div>

          {portfolio && <PortfolioCard portfolio={portfolio} budget={params.budget} />}
        </>
      )}
    </div>
  );
}


function PortfolioCard({ portfolio, budget }: { portfolio: PortfolioOut; budget: number }) {
  const m = portfolio.metrics;
  const fmt = (n: number) => n.toLocaleString("en-US", { style: "currency", currency: "USD", maximumFractionDigits: 0 });
  const pct = (n: number) => `${n.toFixed(2)}%`;
  const shortBookAlpha =
    portfolio.shorts.length
      ? (
          portfolio.shorts.reduce((acc, p) => acc + (p.alpha_pct ?? 0) * p.weight, 0) /
          portfolio.shorts.reduce((acc, p) => acc + p.weight, 0)
        ).toFixed(2) + "%"
      : "—"

  return (
    <div className="portfolio-card">
      <div className="metrics-grid">
        <Metric label="Target Exposure"    value={pct(m.target_exposure_pct)}  highlight />
        <Metric label="Non-Target Long"    value={pct(m.non_target_long_pct)} />
        <Metric label="Short Book"         value={pct(m.short_total_pct)} />
        <Metric label="Net Non-Target"     value={pct(m.net_non_target_pct)} />
        <Metric label="Unhedgeable"        value={pct(m.unhedgeable_pct)} />
        <Metric label="Gross Leverage"     value={m.gross_leverage.toFixed(2) + "x"} />
        <Metric label="Annual Carry Cost"  value={pct(m.annual_carry_cost_pct)} />
        <Metric label="Short Book Alpha"   value={shortBookAlpha} />
      </div>

      <div className="positions-grid">
        <PositionsTable
          title={`Longs (${m.long_count})`}
          positions={portfolio.longs}
          fmt={fmt}
        />
        <PositionsTable
          title={`Shorts (${m.short_count})`}
          positions={portfolio.shorts}
          fmt={fmt}
        />
      </div>

      {portfolio.unhedgeable_names.length > 0 && (
        <details className="unhedgeable-section">
          <summary>Unhedgeable holdings ({portfolio.unhedgeable_names.length})</summary>
          <p className="unhedgeable-note">
            Private, restricted, or unlisted — cannot be shorted.
          </p>
          <ul>
            {portfolio.unhedgeable_names.map((n, i) => <li key={i}>{n}</li>)}
          </ul>
        </details>
      )}
    </div>
  );
}


function Metric({ label, value, highlight = false }: { label: string; value: string; highlight?: boolean }) {
  return (
    <div className={`metric ${highlight ? "metric-highlight" : ""}`}>
      <span className="metric-label">{label}</span>
      <span className="metric-value">{value}</span>
    </div>
  );
}


function PositionsTable({
  title, positions, fmt,
}: {
  title: string;
  positions: PositionOut[];
  fmt: (n: number) => string;
}) {
  if (positions.length === 0) return null;
  return (
    <div className="positions-table">
      <h4>{title}</h4>
      <table>
        <thead>
          <tr>
            <th>Name</th>
            <th>ID</th>
            <th className="num">Weight</th>
            <th className="num">Amount</th>
            <th className="num">Cost/yr</th>
            <th className="num">Alpha</th>
          </tr>
        </thead>
        <tbody>
          {positions.map((p, i) => (
            <tr key={i}>
              <td>{p.name}</td>
              <td>{p.identifier ?? "—"}</td>
              <td className="num">{(p.weight * 100).toFixed(1)}%</td>
              <td className="num">{fmt(p.dollar_amount)}</td>
              <td className="num">{p.annual_cost_pct.toFixed(2)}%</td>
              <td className={p.alpha_pct && p.alpha_pct < 0 ? "negative" : "positive"}>
                {p.alpha_pct != null ? (p.alpha_pct > 0 ? "+" : "") + p.alpha_pct.toFixed(2) + "%" : "—"}
              </td>
            </tr>
          ))}
        </tbody>
      </table>
    </div>
  );
}