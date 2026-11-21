/* Shared style maps and formatting helpers for the command centre. */

export const RISK_STYLES = {
  VERY_HIGH: { label: "Very high", chip: "bg-red-100 text-red-800 ring-red-200", color: "#dc2626" },
  HIGH: { label: "High", chip: "bg-orange-100 text-orange-800 ring-orange-200", color: "#ea580c" },
  MODERATE: { label: "Moderate", chip: "bg-amber-100 text-amber-800 ring-amber-200", color: "#d97706" },
  LOW: { label: "Low", chip: "bg-emerald-100 text-emerald-800 ring-emerald-200", color: "#059669" },
  UNKNOWN: { label: "Unknown", chip: "bg-slate-100 text-slate-600 ring-slate-200", color: "#94a3b8" },
};

export const SEVERITY_STYLES = {
  CRITICAL: "bg-red-600 text-white",
  HIGH: "bg-orange-500 text-white",
  MEDIUM: "bg-amber-400 text-amber-950",
  LOW: "bg-slate-300 text-slate-800",
};

export const STOCK_STYLES = {
  STOCKOUT: "bg-red-600 text-white",
  CRITICAL: "bg-red-100 text-red-800",
  LOW: "bg-amber-100 text-amber-800",
  OK: "bg-emerald-50 text-emerald-700",
  SURPLUS: "bg-sky-50 text-sky-700",
};

export const SIGNAL_STYLES = {
  high: "bg-red-100 text-red-800",
  elevated: "bg-amber-100 text-amber-800",
  normal: "bg-slate-100 text-slate-600",
};

export function fmt(value, digits = 0) {
  if (value === null || value === undefined || Number.isNaN(value)) return "—";
  return Number(value).toLocaleString(undefined, { maximumFractionDigits: digits, minimumFractionDigits: digits });
}

export function shortDate(iso) {
  if (!iso) return "—";
  return new Date(`${iso.slice(0, 10)}T00:00:00`).toLocaleDateString(undefined, { day: "numeric", month: "short" });
}
