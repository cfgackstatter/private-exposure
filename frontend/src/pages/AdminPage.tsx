import { useCallback, useEffect, useState } from "react";
import { api } from "../api/client";
import { IngestPanel } from "../components/IngestPanel";
import { FundsTable } from "../components/FundsTable";
import type { FundOut } from "../types";

export function AdminPage() {
  const [funds, setFunds] = useState<FundOut[]>([]);

  const loadFunds = useCallback(() => {
    api.listFunds().then(setFunds).catch(console.error);
  }, []);

  useEffect(() => { loadFunds(); }, [loadFunds]);

  return (
    <div className="page">
      <header className="page-header">
        <div className="logo">
          <svg width="24" height="24" viewBox="0 0 24 24" fill="none" aria-label="Private Exposure">
            <circle cx="12" cy="12" r="10" stroke="currentColor" strokeWidth="1.5"/>
            <circle cx="12" cy="12" r="4" fill="currentColor"/>
            <line x1="12" y1="2" x2="12" y2="8" stroke="currentColor" strokeWidth="1.5"/>
            <line x1="12" y1="16" x2="12" y2="22" stroke="currentColor" strokeWidth="1.5"/>
            <line x1="2" y1="12" x2="8" y2="12" stroke="currentColor" strokeWidth="1.5"/>
            <line x1="16" y1="12" x2="22" y2="12" stroke="currentColor" strokeWidth="1.5"/>
          </svg>
          <span>Private Exposure</span>
        </div>
        <span className="admin-badge">Admin</span>
      </header>
      <main className="page-main">
        <IngestPanel onIngested={loadFunds} />
        <FundsTable funds={funds} onDeleted={loadFunds} />
      </main>
    </div>
  );
}