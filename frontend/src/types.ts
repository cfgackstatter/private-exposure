export interface HoldingOut {
  issuer_name: string;
  title: string | null;
  cusip: string | null;
  isin: string | null;
  ticker: string | null;
  lei: string | null;
  other_id: string | null;
  other_id_desc: string | null;
  asset_category: string | null;
  issuer_type: string | null;
  value_usd: number;
  pct_of_net_assets: number;
  quantity: number;
  quantity_units: string | null;
  currency: string | null;
  country: string | null;
  payoff_profile: string | null;
  fair_value_level: string | null;
  is_restricted: boolean | null;
  is_cash_collateral: boolean | null;
  is_non_cash_collateral: boolean | null;
  is_loan_by_fund: boolean | null;
  maturity_date: string | null;
  coupon_rate: number | null;
  is_default: boolean | null;
  derivative_category: string | null;
  notional_amount: number | null;
}

export interface FilingOut {
  id: number;
  accession_no: string;
  form_type: string;
  filing_date: string;
  report_date: string;
  holdings: HoldingOut[];
}

export interface FundOut {
  id: number;
  cik: number;
  ticker: string;
  series_id: string;
  company_name: string;
  series_name: string;
  class_name: string;
  filings: FilingOut[];
}

export interface IngestResult {
  ticker: string;
  dates_ingested: string[];
}

export interface MatchedHolding {
  issuer_name: string;
  title: string | null;
  ticker: string | null;
  isin: string | null;
  cusip: string | null;
  asset_category: string | null;
  country: string | null;
  pct_of_net_assets: number;
}

export interface FundSearchResult {
  id: number
  ticker: string
  series_name: string
  class_name: string
  cik: number
  total_weight: number
  match_count: number
  matched_holdings: MatchedHolding[]
  report_date: string
}

export interface OptimizeRequest {
  keywords: string[];
  budget: number;
  max_gross_leverage: number;
  max_fund_weight: number | null;
  borrow_cost: number;
  financing_cost: number;
  holding_period_months: number;
  max_funds: number;
  max_hedges: number;
  short_cap: number;
}

export interface PositionOut {
  name: string;
  identifier: string | null;
  weight: number;
  dollar_amount: number;
  annual_cost_pct: number;
  alpha_pct: number | null;
}

export interface PortfolioMetricsOut {
  target_exposure_pct: number;
  non_target_long_pct: number;
  short_total_pct: number;
  net_non_target_pct: number;
  unhedgeable_pct: number;
  gross_leverage: number;
  annual_carry_cost_pct: number;
  long_count: number;
  short_count: number;
}

export interface PortfolioOut {
  label: string;
  metrics: PortfolioMetricsOut;
  longs: PositionOut[];
  shorts: PositionOut[];
  unhedgeable_names: string[];
}

export interface OptimizeResponse {
  keywords: string[];
  portfolios: PortfolioOut[];
  warnings: string[];
}