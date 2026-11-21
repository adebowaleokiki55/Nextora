import { useCallback, useEffect, useMemo, useState } from "react";
import { Activity, Database, Microscope, RefreshCw } from "lucide-react";
import "./dashboard.css";
import { apiGet, apiPost } from "./api";
import AlertsPanel from "./AlertsPanel";
import ApprovalQueue from "./ApprovalQueue";
import { EvaluationPanel, LabPanel, SourcesPanel } from "./InfoPanels";
import LocationPanel from "./LocationPanel";
import RiskMap from "./RiskMap";
import RunPanel from "./RunPanel";
import { fmt } from "./format";
import { Button, ErrorNote, RiskChip } from "./ui";

async function settle(promises) {
  const results = await Promise.allSettled(promises);
  return results.map((r) => (r.status === "fulfilled" ? r.value : null));
}

function Stat({ label, value, hint, tone = "text-slate-900" }) {
  return (
    <div className="rounded-2xl bg-white px-4 py-3 shadow-sm ring-1 ring-slate-200">
      <div className="text-[11px] font-semibold uppercase tracking-wide text-slate-500">{label}</div>
      <div className={`mt-1 text-2xl font-bold ${tone}`}>{value}</div>
      {hint && <div className="mt-0.5 truncate text-[11px] text-slate-500">{hint}</div>}
    </div>
  );
}

export default function Dashboard() {
  const [data, setData] = useState({
    health: null, risk: [], alerts: [], proposals: [], shortages: [], facilities: [],
    labPending: 0, evaluation: null, sources: [], modelCard: null,
  });
  const [selectedId, setSelectedId] = useState(null);
  const [detail, setDetail] = useState({ prediction: null, surveillance: null, stock: null });
  const [version, setVersion] = useState(0);
  const [error, setError] = useState("");
  const [seeding, setSeeding] = useState(false);

  const refresh = useCallback(() => setVersion((v) => v + 1), []);

  useEffect(() => {
    let cancelled = false;
    settle([
      apiGet("/api/health"),
      apiGet("/api/prediction/risk"),
      apiGet("/api/alerts"),
      apiGet("/api/proposals"),
      apiGet("/api/supply/shortages"),
      apiGet("/api/facilities"),
      apiGet("/api/lab/submissions?status=PENDING_REVIEW"),
      apiGet("/api/prediction/evaluation"),
      apiGet("/api/surveillance/sources"),
      apiGet("/api/lab/model-card"),
    ]).then(([health, risk, alerts, proposals, shortages, facilities, lab, evaluation, sources, modelCard]) => {
      if (cancelled) return;
      if (!health) {
        setError("Cannot reach the Nextora backend. Start it with: python -m uvicorn backend.main:app --port 8000");
        return;
      }
      setError("");
      const riskRows = risk?.locations || [];
      setData({
        health,
        risk: riskRows,
        alerts: alerts?.alerts || [],
        proposals: proposals?.proposals || [],
        shortages: shortages?.shortages || [],
        facilities: facilities?.facilities || [],
        labPending: lab?.submissions?.length || 0,
        evaluation,
        sources: sources?.sources || [],
        modelCard,
      });
      setSelectedId((current) =>
        current && riskRows.some((r) => r.location.id === current) ? current : riskRows[0]?.location.id ?? null
      );
    });
    return () => {
      cancelled = true;
    };
  }, [version]);

  useEffect(() => {
    if (!selectedId) return undefined;
    let cancelled = false;
    settle([
      apiGet(`/api/prediction/locations/${selectedId}`),
      apiGet(`/api/surveillance/locations/${selectedId}?weeks=52`),
      apiGet(`/api/supply/status?location_id=${selectedId}`),
    ]).then(([prediction, surveillance, stock]) => {
      if (!cancelled) setDetail({ prediction, surveillance, stock: stock?.stock || [] });
    });
    return () => {
      cancelled = true;
    };
  }, [selectedId, version]);

  async function seed() {
    const ok = window.confirm(
      "Reset the demo? This deletes ALL scenario data (lab submissions, approvals, agent runs, stock changes) and reloads the synthetic Lagos scenario. It takes about 20 seconds."
    );
    if (!ok) return;
    setSeeding(true);
    setError("");
    try {
      await apiPost("/api/demo/seed");
      setSelectedId(null);
      refresh();
    } catch (err) {
      setError(err.message);
    } finally {
      setSeeding(false);
    }
  }

  const shortageFacilityIds = useMemo(() => new Set(data.shortages.map((s) => s.facility.id)), [data.shortages]);
  const selectedRisk = data.risk.find((row) => row.location.id === selectedId);
  const warnings = data.risk.filter((row) => row.early_warning);
  const top = data.risk[0];
  const critical = data.alerts.filter((a) => a.severity === "CRITICAL").length;
  const pending = data.proposals.filter((p) => p.status === "PENDING").length;

  return (
    <div className="nx-dash">
      <header className="sticky top-0 z-[1100] border-b border-slate-200 bg-white/95 backdrop-blur">
        <div className="mx-auto flex max-w-[1500px] flex-wrap items-center justify-between gap-3 px-5 py-3">
          <a href="/" className="flex items-center gap-2 no-underline">
            <span className="flex h-9 w-9 items-center justify-center rounded-xl bg-emerald-700 text-white">
              <Microscope size={19} />
            </span>
            <span>
              <span className="block text-sm font-extrabold text-slate-900">nextora<span className="text-emerald-600">.</span></span>
              <span className="block text-[10px] font-semibold tracking-widest text-slate-500">MALARIA COMMAND CENTRE</span>
            </span>
          </a>
          <nav className="flex flex-wrap items-center gap-1 text-xs font-semibold text-slate-600">
            <a className="rounded-lg px-2 py-1 no-underline hover:bg-slate-100" href="/?view=analyzer">Cell Explorer</a>
            <a className="rounded-lg px-2 py-1 no-underline hover:bg-slate-100" href="/?view=lab-review">Lab review</a>
            <a className="rounded-lg px-2 py-1 no-underline hover:bg-slate-100" href="/?view=treatment">Treatment support</a>
            <Button variant="secondary" onClick={refresh}><RefreshCw size={13} /> Refresh</Button>
            <Button variant="danger" onClick={seed} disabled={seeding}>
              <Database size={13} /> {seeding ? "Seeding…" : "Seed demo data"}
            </Button>
          </nav>
        </div>
      </header>

      <main className="mx-auto max-w-[1500px] space-y-4 px-5 py-4">
        <div className="flex flex-wrap items-center gap-2 rounded-xl border border-dashed border-amber-400 bg-amber-50 px-4 py-2 text-[11px] text-amber-900">
          <Activity size={14} />
          <strong>Demo scenario.</strong>
          Malaria case counts, facilities and stock are SYNTHETIC; weather is REAL reanalysis data. Nextora is a
          decision-support prototype for public-health teams, not an official surveillance or diagnostic system.
        </div>

        <ErrorNote error={error} />

        <div className="grid grid-cols-2 gap-3 md:grid-cols-3 xl:grid-cols-6">
          <Stat label="Highest risk" value={top ? top.location.name : "—"} hint={top && <RiskChip level={top.risk_level} score={top.risk_score} />} />
          <Stat label="Early warnings" value={warnings.length} hint={warnings.map((w) => w.location.name).join(", ") || "None"} tone={warnings.length ? "text-red-700" : undefined} />
          <Stat label="Unresolved alerts" value={data.alerts.length} hint={`${critical} critical`} tone={critical ? "text-red-700" : undefined} />
          <Stat label="Supply shortages" value={data.shortages.length} hint={data.shortages.map((s) => s.facility.name.replace(" Demo Health Centre", "")).join(", ") || "None"} tone={data.shortages.length ? "text-orange-700" : undefined} />
          <Stat label="Awaiting approval" value={pending} hint="Agent proposals" tone={pending ? "text-indigo-700" : undefined} />
          <Stat label="Lab results to review" value={data.labPending} hint="Pending qualified review" />
        </div>

        <div className="grid gap-4 xl:grid-cols-12">
          <div className="space-y-4 xl:col-span-7">
            <section className="rounded-2xl bg-white p-3 shadow-sm ring-1 ring-slate-200">
              <div className="mb-2 flex items-center justify-between px-1">
                <h2 className="text-sm font-bold text-slate-900">Lagos: forecast malaria risk by LGA</h2>
                <span className="text-[11px] text-slate-500">Click a circle to inspect a location</span>
              </div>
              <RiskMap
                risk={data.risk}
                facilities={data.facilities}
                shortageFacilityIds={shortageFacilityIds}
                selectedId={selectedId}
                onSelect={setSelectedId}
              />
            </section>
            <LocationPanel
              riskRow={selectedRisk}
              prediction={detail.prediction}
              surveillance={detail.surveillance}
              stock={detail.stock}
            />
          </div>

          <div className="space-y-4 xl:col-span-5">
            <RunPanel
              locations={data.risk}
              selectedId={selectedId}
              onSelect={setSelectedId}
              health={data.health}
              onFinished={refresh}
            />
            <ApprovalQueue proposals={data.proposals} onChanged={refresh} />
            <AlertsPanel alerts={data.alerts} selectedId={selectedId} onChanged={refresh} />
          </div>
        </div>

        <div className="grid gap-4 xl:grid-cols-12">
          <div className="xl:col-span-6"><EvaluationPanel evaluation={data.evaluation} /></div>
          <div className="xl:col-span-3"><SourcesPanel sources={data.sources} /></div>
          <div className="xl:col-span-3"><LabPanel pendingCount={data.labPending} modelCard={data.modelCard} /></div>
        </div>

        <footer className="pb-6 pt-2 text-center text-[11px] text-slate-400">
          Nextora · decision-support research prototype · map data © OpenStreetMap contributors · weather data by Open-Meteo.com ·{" "}
          {fmt(data.risk.length)} LGAs monitored
        </footer>
      </main>
    </div>
  );
}
