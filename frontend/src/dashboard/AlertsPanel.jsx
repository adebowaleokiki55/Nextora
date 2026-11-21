import { useState } from "react";
import { apiPost } from "./api";
import { SEVERITY_STYLES } from "./format";
import { Button, Card, Empty, OriginBadge, Pill } from "./ui";

const MODULE_LABEL = {
  prediction: "Prediction",
  supply: "Supply",
  lab: "Lab",
  surveillance: "Surveillance",
  command: "Command",
};

export default function AlertsPanel({ alerts, selectedId, onChanged }) {
  const [onlySelected, setOnlySelected] = useState(false);
  const shown = onlySelected ? alerts.filter((a) => a.location_id === selectedId) : alerts;

  async function act(alert, action) {
    await apiPost(`/api/alerts/${alert.id}/${action}`);
    onChanged();
  }

  return (
    <Card
      title={`Alerts (${alerts.length})`}
      subtitle="From every module, most severe first"
      actions={
        <label className="flex items-center gap-1 text-[11px] text-slate-500">
          <input type="checkbox" checked={onlySelected} onChange={(e) => setOnlySelected(e.target.checked)} />
          Selected location only
        </label>
      }
    >
      <ul className="max-h-[360px] space-y-2 overflow-auto pr-1">
        {shown.map((alert) => (
          <li key={alert.id} className="rounded-xl p-3 ring-1 ring-slate-200">
            <div className="flex flex-wrap items-center gap-1.5">
              <Pill className={SEVERITY_STYLES[alert.severity]}>{alert.severity}</Pill>
              <Pill className="bg-slate-100 text-slate-600">{MODULE_LABEL[alert.source_module] || alert.source_module}</Pill>
              <OriginBadge origin={alert.data_origin} />
              {alert.status === "ACKNOWLEDGED" && <Pill className="bg-indigo-50 text-indigo-700">Acknowledged</Pill>}
            </div>
            <div className="mt-1 text-xs font-semibold text-slate-800">{alert.title}</div>
            <p className="mt-0.5 text-[11px] leading-snug text-slate-600">{alert.message}</p>
            <div className="mt-2 flex gap-2">
              {alert.status === "OPEN" && (
                <Button variant="secondary" onClick={() => act(alert, "acknowledge")}>Acknowledge</Button>
              )}
              <Button variant="ghost" onClick={() => act(alert, "resolve")}>Resolve</Button>
            </div>
          </li>
        ))}
        {!shown.length && <Empty>No unresolved alerts.</Empty>}
      </ul>
    </Card>
  );
}
