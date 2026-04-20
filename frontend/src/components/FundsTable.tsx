import { useState } from "react";
import type { FundOut } from "../types";
import { HoldingsTable } from "./HoldingsTable";
import { api } from "../api/client";

interface Props {
  funds: FundOut[];
  onDeleted: () => void;
}

interface Selection {
  ticker: string;
  filingId: number;
  reportDate: string;
}

export function FundsTable({ funds, onDeleted }: Props) {
  const [selected, setSelected] = useState<Selection | null>(null);
  const [deleting, setDeleting] = useState<string | null>(null);
  const [confirmTicker, setConfirmTicker] = useState<string | null>(null);

  async function handleDelete(ticker: string) {
    setDeleting(ticker);
    try {
      await api.deleteFund(ticker);
      if (selected?.ticker === ticker) setSelected(null);
      onDeleted();
    } catch (e) {
      console.error(e);
    } finally {
      setDeleting(null);
      setConfirmTicker(null);
    }
  }

  if (funds.length === 0) {
    return <p className="empty">No funds loaded yet. Enter a ticker above to get started.</p>;
  }

  return (
    <div>
      <h2>Loaded Funds</h2>
      {funds.map(fund => (
        <div key={fund.id} className="fund-card">
          <div className="fund-header">
            <span className="fund-ticker">{fund.ticker}</span>
            <span className="fund-name">{fund.series_name}</span>
            <div className="fund-header-right">
              <span className="fund-meta">{fund.class_name} · CIK {fund.cik}</span>
              <div className="fund-actions">
                {confirmTicker === fund.ticker ? (
                  <>
                    <span className="confirm-label">Delete all data?</span>
                    <button
                      className="btn btn-sm btn-danger"
                      onClick={() => handleDelete(fund.ticker)}
                      disabled={deleting === fund.ticker}
                    >
                      {deleting === fund.ticker ? "Deleting…" : "Confirm"}
                    </button>
                    <button
                      className="btn btn-sm btn-ghost"
                      onClick={() => setConfirmTicker(null)}
                      disabled={deleting === fund.ticker}
                    >
                      Cancel
                    </button>
                  </>
                ) : (
                  <button
                    className="btn btn-smbtn-ghost btn-icon"
                    aria-label={`Delete ${fund.ticker}`}
                    onClick={() => setConfirmTicker(fund.ticker)}
                  >
                    <svg width="14" height="14" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round" aria-hidden="true">
                      <polyline points="3 6 5 6 21 6"/>
                      <path d="M19 6l-1 14a2 2 0 0 1-2 2H8a2 2 0 0 1-2-2L5 6"/>
                      <path d="M10 11v6M14 11v6"/>
                      <path d="M9 6V4a1 1 0 0 1 1-1h4a1 1 0 0 1 1 1v2"/>
                    </svg>
                  </button>
                )}
              </div>
            </div>
          </div>

          <div className="table-wrap">
            <table>
              <thead>
                <tr>
                  <th>Report Date</th>
                  <th>Form</th>
                  <th>Filed</th>
                  <th>Accession</th>
                </tr>
              </thead>
              <tbody>
                {fund.filings
                  .sort((a, b) => b.report_date.localeCompare(a.report_date))
                  .map(filing => {
                    const isActive = selected?.filingId === filing.id;
                    return (
                      <tr
                        key={filing.id}
                        className={`filing-row ${isActive ? "active" : ""}`}
                        onClick={() =>
                          setSelected(isActive ? null : {
                            ticker: fund.ticker,
                            filingId: filing.id,
                            reportDate: filing.report_date,
                          })
                        }
                      >
                        <td>{filing.report_date}</td>
                        <td>{filing.form_type}</td>
                        <td>{filing.filing_date}</td>
                        <td className="mono">{filing.accession_no}</td>
                      </tr>
                    );
                  })}
              </tbody>
            </table>
          </div>

          {selected?.ticker === fund.ticker && (
            <HoldingsTable
              ticker={selected.ticker}
              filingId={selected.filingId}
              reportDate={selected.reportDate}
            />
          )}
        </div>
      ))}
    </div>
  );
}