import { fmt, shortDate } from "./format";
import { Card, Empty, OriginBadge } from "./ui";

export function EvaluationPanel({ evaluation }) {
  if (!evaluation) return <Card title="Forecast accuracy"><Empty>No evaluation available.</Empty></Card>;
  const rows = Object.values(evaluation.horizons || {});

  return (
    <Card
      title="Forecast accuracy vs baselines"
      subtitle={`Held-out test period (${evaluation.split?.test_target_weeks}) · model ${evaluation.model_version}`}
      actions={<OriginBadge origin={evaluation.case_data_origin} />}
    >
      <div className="overflow-auto">
        <table className="w-full text-left text-[11px]">
          <thead className="text-slate-500">
            <tr>
              <th className="py-1 pr-2 font-semibold">Weeks ahead</th>
              <th className="py-1 pr-2 text-right font-semibold">Model MAE</th>
              <th className="py-1 pr-2 text-right font-semibold">Last week</th>
              <th className="py-1 pr-2 text-right font-semibold">Same week last year</th>
              <th className="py-1 pr-2 text-right font-semibold">Usual for week</th>
              <th className="py-1 pr-2 text-right font-semibold">Alert precision / recall</th>
              <th className="py-1 text-right font-semibold">Last week: precision / recall</th>
            </tr>
          </thead>
          <tbody>
            {rows.map((r) => {
              const best = Math.min(r.model.mae, r.baseline_persistence.mae, r.baseline_seasonal_naive.mae, r.baseline_hist_mean.mae);
              const cell = (value) => (
                <td className={`py-1 pr-2 text-right tabular-nums ${value === best ? "font-bold text-emerald-700" : ""}`}>{fmt(value, 1)}</td>
              );
              const pr = (c) => `${fmt(c.precision, 2)} / ${fmt(c.recall, 2)}`;
              return (
                <tr key={r.horizon_weeks} className="border-t border-slate-100">
                  <td className="py-1 pr-2">{r.horizon_weeks}</td>
                  {cell(r.model.mae)}
                  {cell(r.baseline_persistence.mae)}
                  {cell(r.baseline_seasonal_naive.mae)}
                  {cell(r.baseline_hist_mean.mae)}
                  <td className="py-1 pr-2 text-right tabular-nums">{pr(r.model.crossing)}</td>
                  <td className="py-1 text-right tabular-nums">{pr(r.baseline_persistence.crossing)}</td>
                </tr>
              );
            })}
          </tbody>
        </table>
      </div>
      <p className="mt-2 text-[11px] leading-snug text-slate-500">
        MAE = mean absolute error in cases per week (lower is better; best in green).
        {rows[0] && ` ${rows[0].model.crossing.actual_crossings} threshold crossings in the test period.`}
      </p>
      <p className="mt-2 rounded-lg bg-amber-50 px-3 py-2 text-[11px] leading-snug text-amber-900">{evaluation.honesty_note}</p>
    </Card>
  );
}

export function SourcesPanel({ sources }) {
  return (
    <Card title="Data sources" subtitle="Where every dataset comes from, and whether it is real">
      <ul className="space-y-2">
        {sources.map((s) => (
          <li key={s.metric} className="text-[11px]">
            <div className="flex items-center gap-2">
              <span className="font-semibold text-slate-800">{s.label}</span>
              <OriginBadge origin={s.data_origin} />
            </div>
            <p className="leading-snug text-slate-500">
              {s.sources[0]} · {s.locations} LGAs · {shortDate(s.first_week)} {s.first_week?.slice(0, 4)} – {shortDate(s.last_week)} {s.last_week?.slice(0, 4)}
            </p>
          </li>
        ))}
        {!sources.length && <Empty>No data loaded.</Empty>}
      </ul>
      <p className="mt-3 text-[11px] text-slate-500">
        Facilities and stock are fictional demo data (SYNTHETIC). LGA names are real; their map points are approximate.
      </p>
    </Card>
  );
}

export function LabPanel({ pendingCount, modelCard }) {
  return (
    <Card title="Laboratory intelligence" subtitle="Blood-smear decision support (Module 5)">
      <div className="flex items-baseline gap-2">
        <span className="text-2xl font-bold text-slate-900">{pendingCount}</span>
        <span className="text-xs text-slate-500">submission(s) awaiting qualified review</span>
      </div>
      <p className="mt-2 text-[11px] leading-snug text-slate-500">
        Only results verified by a qualified laboratory professional feed surveillance. Not a diagnostic tool and
        not a validated parasite counter.
      </p>
      {modelCard && (
        <p className="mt-2 text-[11px] leading-snug text-slate-500">
          {modelCard.model_name} ({modelCard.model_version}), trained on the NIH Malaria Cell Images dataset.{" "}
          {modelCard.metrics ? "" : "Held-out metrics: not yet available."}
        </p>
      )}
      <div className="mt-3 flex flex-wrap gap-2 text-xs font-semibold">
        <a className="rounded-lg bg-emerald-700 px-3 py-1.5 text-white no-underline" href="/?view=lab-review">Review queue</a>
        <a className="rounded-lg px-3 py-1.5 text-slate-700 no-underline ring-1 ring-slate-300" href="/?view=analyzer">Cell Explorer</a>
      </div>
    </Card>
  );
}
