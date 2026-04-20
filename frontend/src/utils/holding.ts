// src/utils/holding.ts
import type { HoldingOut } from "../types";
import type { MatchedHolding } from "../types";

export function identifier(h: Pick<HoldingOut | MatchedHolding, "ticker" | "isin" | "cusip">): string {
  return h.ticker ?? h.isin ?? h.cusip ?? "—";
}