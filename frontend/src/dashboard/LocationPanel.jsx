import { useMemo } from "react";
import Plot from "./Plot";
import { SIGNAL_STYLES, STOCK_STYLES, fmt, shortDate } from "./format";
import { Card, Empty, OriginBadge, Pill, RiskChip } from "./ui";

function ForecastChart({ prediction }) {
  const { data, layout } = useMemo(() => {
    const history = prediction.history || [];
    const forecasts = prediction.forecasts || [];
    const lastActual = history[history.length - 1];
    const since = history[0]?.week_start;
    const backtest = (prediction.backtest || []).filter((row) => row.horizon === 1 && (!since || row.target_week >= since));

    const fx = [lastActual?.week_start, ...forecasts.map((f) => f.target_week)].filter(Boolean);
    const fy = [lastActual?.cases, ...forecasts.map((f) => f.predicted_cases)];
    const traces = [
      {
        x: history.map((h) => h.week_start),
        y: history.map((h) => h.cases),
        name: "Reported cases",
        mode: "lines+markers",
        line: { color: "#0f172a", width: 2 },
        marker: { size: 4 },
      },
      {
        x: backtest.map((row) => row.target_week),
        y: backtest.map((row) => row.predicted),
        name: "Past 1-week forecasts (backtest)",
        mode: "lines",
        line: { color: "#94a3b8", width: 1.5, dash: "dot" },
      },
      {
        x: forecasts.map((f) => f.target_week),
        y: forecasts.map((f) => f.upper_cases),
        mode: "lines",
        line: { width: 0 },
        showlegend: false,
        hoverinfo: "skip",
      },
      {
        x: forecasts.map((f) => f.target_week),
        y: forecasts.map((f) => f.lower_cases),
        name: "80% range",
        mode: "lines",
        fill: "tonexty",
        fillcolor: "rgba(234, 88, 12, 0.15)",
        line: { width: 0 },
      },
      {
        x: fx,
        y: fy,
        name: "Forecast",
        mode: "lines+markers",
        line: { color: "#ea580c", width: 2.5, dash: "dash" },
        marker: { size: 6 },
      },
      {
        x: forecasts.map((f) => f.target_week),
        y: forecasts.map((f) => f.alert_threshold),
        name: "Alert threshold",
        mode: "lines+markers",
        line: { color: "#dc2626", width: 1.5, dash: "dot" },
        marker: { symbol: "line-ew-open", size: 10 },
      },
    ];
    return {
      data: traces,
      layout: {
        yaxis: { title: { text: "Cases / week" }, rangemode: "tozero" },
        legend: { orientation: "h", x: 0, y: 1.18, font: { size: 10 } },
        margin: { l: 44, r: 12, t: 40, b: 30 },
      },
    };
  }, [prediction]);

  return <Plot data={data} layout={layout} height={270} />;
}

function WeatherChart({ surveillance }) {
  const { data, layout } = useMemo(() => {
    const rain = surveillance.series?.rainfall_mm?.points || [];
    const cases = surveillance.series?.malaria_cases_confirmed?.points || [];
    return {
      data: [
        {
          type: "bar",
          x: rain.map((p) => p.week_start),
          y: rain.map((p) => p.value),
          name: "Rainfall (mm/week)",
          marker: { color: "#7dd3fc" },
        },
        {
          x: cases.map((p) => p.week_start),
          y: cases.map((p) => p.value),
          name: "Cases / week",
          yaxis: "y2",
          mode: "lines",
          line: { color: "#0f172a", width: 1.5 },
        },
      ],
      layout: {
        yaxis: { title: { text: "Rainfall (mm)" }, rangemode: "tozero" },
        yaxis2: { title: { text: "Cases" }, overlaying: "y", side: "right", rangemode: "tozero", showgrid: false },
        margin: { l: 44, r: 44, t: 8, b: 36 },
      },
    };
  }, [surveillance]);

  return <Plot data={data} layout={layout} height={220} />;
}

export default function LocationPanel({ riskRow, prediction, surveillance, stock }) {
  if (!riskRow) return <Card title="Location"><Empty>Select a location on the map.</Empty></Card>;

  const signals = surveillance?.signals || [];
  const casesOrigin = prediction?.data_origin || surveillance?.series?.malaria_cases_confirmed?.data_origin;
  const weatherOrigin = surveillance?.series?.rainfall_mm?.data_origin;

  return (
    <Card
      title={`${riskRow.location.name}`}
      subtitle={`Forecast made with data up to the week of ${shortDate(riskRow.forecasts?.[0]?.origin_week)} · model ${riskRow.model_version || "—"}`}
      actions={<RiskChip level={riskRow.risk_level} score={riskRow.risk_score} />}
    >
      <div className="grid gap-4 lg:grid-cols-5">
        <div className="lg:col-span-3">
          <div className="mb-1 flex items-center gap-2 text-xs font-semibold text-slate-600">
            Cases and 1–4 week forecast <OriginBadge origin={casesOrigin} />
          </div>
          {prediction ? <ForecastChart prediction={prediction} /> : <Empty>Loading forecast…</Empty>}
          <p className="mt-1 text-[11px] leading-snug text-slate-500">
            Risk score = highest estimated chance (×100) that weekly cases reach the alert threshold
            (mean + 2 SD of the same weeks in earlier years) within 4 weeks.
            {riskRow.early_warning && (
              <strong className="text-red-700"> Early warning active from the week of {shortDate(riskRow.first_warning_week)}.</strong>
            )}
          </p>
        </div>

        <div className="lg:col-span-2">
          <div className="mb-2 text-xs font-semibold text-slate-600">Surveillance signals</div>
          <ul className="space-y-2">
            {signals.map((signal) => (
              <li key={signal.key} className="rounded-lg bg-slate-50 px-3 py-2">
                <div className="flex items-center justify-between gap-2">
                  <span className="text-xs font-semibold text-slate-800">{signal.label}</span>
                  <span className="flex items-center gap-1">
                    <OriginBadge origin={signal.data_origin} />
                    <Pill className={SIGNAL_STYLES[signal.status]}>{signal.status}</Pill>
                  </span>
                </div>
                <p className="mt-1 text-[11px] leading-snug text-slate-600">{signal.explanation}</p>
              </li>
            ))}
            {!signals.length && <Empty>No signals.</Empty>}
          </ul>
        </div>
      </div>

      <div className="mt-4 grid gap-4 lg:grid-cols-2">
        <div>
          <div className="mb-1 flex flex-wrap items-center gap-2 text-xs font-semibold text-slate-600">
            Rainfall and cases, last 52 weeks <OriginBadge origin={weatherOrigin} />
            <span className="font-normal text-slate-400">rain</span>
            <OriginBadge origin={casesOrigin} />
            <span className="font-normal text-slate-400">cases</span>
          </div>
          {surveillance ? <WeatherChart surveillance={surveillance} /> : <Empty>Loading…</Empty>}
          {weatherOrigin === "REAL" && (
            <p className="text-[10px] text-slate-400">Weather data by Open-Meteo.com (ERA5 reanalysis, CC BY 4.0).</p>
          )}
        </div>

        <div>
          <div className="mb-2 flex items-center gap-2 text-xs font-semibold text-slate-600">
            Facility stock <OriginBadge origin="SYNTHETIC" />
          </div>
          <div className="max-h-[230px] overflow-auto rounded-lg ring-1 ring-slate-100">
            <table className="w-full text-left text-[11px]">
              <thead className="sticky top-0 bg-slate-50 text-slate-500">
                <tr>
                  <th className="px-2 py-1.5 font-semibold">Facility</th>
                  <th className="px-2 py-1.5 font-semibold">Item</th>
                  <th className="whitespace-nowrap px-2 py-1.5 text-right font-semibold">On hand</th>
                  <th className="px-2 py-1.5 text-right font-semibold">Weeks</th>
                  <th className="px-2 py-1.5 font-semibold">Status</th>
                </tr>
              </thead>
              <tbody>
                {(stock || []).map((row) => (
                  <tr key={`${row.facility.id}-${row.item.code}`} className="border-t border-slate-100">
                    <td className="px-2 py-1.5" title={row.facility.name}>
                      {row.facility.name
                        .replace(`${riskRow.location.name} `, "")
                        .replace("Demo Health Centre", "Centre")
                        .replace("Demo Laboratory", "Laboratory")}
                    </td>
                    <td className="px-2 py-1.5">{row.item.code}</td>
                    <td className="px-2 py-1.5 text-right tabular-nums">{fmt(row.on_hand)}</td>
                    <td className="px-2 py-1.5 text-right tabular-nums">{fmt(row.weeks_of_stock, 1)}</td>
                    <td className="px-2 py-1.5">
                      <Pill className={STOCK_STYLES[row.status]}>{row.status}</Pill>
                    </td>
                  </tr>
                ))}
              </tbody>
            </table>
            {!stock?.length && <Empty>No stock records.</Empty>}
          </div>
        </div>
      </div>
    </Card>
  );
}
