const BASE = "http://localhost:8000";

async function request<T>(path: string, options?: RequestInit): Promise<T> {
  const res = await fetch(`${BASE}${path}`, options);
  if (!res.ok) {
    const body = await res.json().catch(() => ({}));
    throw new Error(body.detail ?? `HTTP ${res.status}`);
  }
  if (res.status === 204) return undefined as T;
  return res.json();
}

export const api = {
  ingestFund: (ticker: string, numQuarters = 4, cik?: number, seriesId?: string) =>
    request<{ ticker: string; dates_ingested: string[] }>(
      `/admin/funds/${ticker}`,
      {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({
          num_quarters: numQuarters,
          cik: cik ?? null,
          series_id: seriesId ?? null,
        }),
      }
    ),

  listFunds: () =>
    request<import("../types").FundOut[]>("/admin/funds"),

  getHoldings: (ticker: string, filingId: number) =>
    request<import("../types").HoldingOut[]>(
      `/admin/funds/${ticker}/filings/${filingId}/holdings`
    ),

  deleteFund: (ticker: string) =>
    request<void>(`/admin/funds/${ticker}`, { method: "DELETE" }),

  search: (q: string) =>
    request<import("../types").FundSearchResult[]>(
      `/search?${new URLSearchParams({ q })}`
    ),

  optimize: (req: import("../types").OptimizeRequest) =>
    request<import("../types").OptimizeResponse>("/optimize", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify(req),
    }),
};