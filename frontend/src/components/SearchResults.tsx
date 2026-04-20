import { useState, useEffect } from "react"
import type { FundSearchResult } from "../types"
import { identifier } from "../utils/holding";

interface Props {
  results: FundSearchResult[]
  query: string
}

export function SearchResults({ results, query }: Props) {
  const [expanded, setExpanded] = useState<number | null>(null)

  // reset expanded when results change
  useEffect(() => {
    setExpanded(null)
  }, [results])
  
  if (results.length === 0) {
    return <p className="empty">No funds matched <strong>{query}</strong>.</p>
  }

  return (
    <div>
      <p className="search-meta">
        {results.length} fund{results.length !== 1 ? "s" : ""} matched &nbsp;·&nbsp;
        sorted by matched weight
      </p>
      {results.map(r => (
        <div key={r.id} className="fund-card">
          <div
            className="fund-header search-result-header"
            onClick={() => setExpanded(expanded === r.id ? null : r.id)}
            style={{ cursor: "pointer" }}
          >
            <span className="fund-ticker">{r.ticker}</span>
            <span className="fund-name">{r.series_name}</span>
            <div className="fund-header-right">
              <span className="fund-meta">{r.class_name} · as of {r.report_date}</span>
              <div className="match-bar-wrap">
                <div
                  className="match-bar"
                  style={{ width: `${r.total_weight}%` }}
                />
              </div>
              <span className="match-weight">{r.total_weight.toFixed(2)}%</span>
            </div>
          </div>

          {/* top holdings preview always visible */}
          <div className="match-chips">
            {r.matched_holdings.slice(0, 5).map((h, i) => (
              <span key={i} className="match-chip">
                {h.title ?? h.issuer_name} &nbsp;
                <span className="chip-weight">{h.pct_of_net_assets.toFixed(2)}%</span>
              </span>
            ))}
            {r.matched_holdings.length > 5 && (
              <span className="match-chip chip-more">+{r.matched_holdings.length - 5} more</span>
            )}
          </div>

          {/* expanded full matched holdings */}
          {expanded === r.id && (
            <div className="matched-holdings-detail">
              <h3 className="holdings-section-title">{r.match_count} matched holdings</h3>
              <div className="table-wrap">
                <table>
                  <thead>
                    <tr>
                      <th>%</th>
                      <th>Name</th>
                      <th>Identifier</th>
                      <th>Category</th>
                      <th>Country</th>
                    </tr>
                  </thead>
                  <tbody>
                    {r.matched_holdings.map((h, i) => (
                      <tr key={i}>
                        <td className="num">{h.pct_of_net_assets.toFixed(2)}%</td>
                        <td title={h.issuer_name}>{h.title ?? h.issuer_name}</td>
                        <td>{identifier(h)}</td>
                        <td>{h.asset_category ?? "—"}</td>
                        <td>{h.country ?? "—"}</td>
                      </tr>
                    ))}
                  </tbody>
                </table>
              </div>
            </div>
          )}
        </div>
      ))}
    </div>
  )
}