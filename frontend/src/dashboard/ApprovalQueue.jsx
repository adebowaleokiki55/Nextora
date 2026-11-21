import { useState } from "react";
import { apiPost } from "./api";
import { fmt } from "./format";
import { Button, Card, Empty, ErrorNote, Pill } from "./ui";

const TYPE_STYLES = {
  STOCK_TRANSFER: "bg-violet-100 text-violet-800",
  REPLENISHMENT_ORDER: "bg-sky-100 text-sky-800",
  PUBLIC_MESSAGE: "bg-pink-100 text-pink-800",
};

const STATUS_STYLES = {
  EXECUTED: "bg-emerald-100 text-emerald-800",
  FAILED: "bg-red-100 text-red-800",
  REJECTED: "bg-slate-200 text-slate-700",
  APPROVED: "bg-emerald-50 text-emerald-700",
};

function Details({ proposal }) {
  const p = proposal.payload;
  if (proposal.proposal_type === "STOCK_TRANSFER") {
    return (
      <p className="text-[11px] text-slate-600">
        <strong>{fmt(p.quantity)} {p.item_code}</strong> · {p.from_facility_name} → {p.to_facility_name} · {fmt(p.distance_km, 1)} km
      </p>
    );
  }
  if (proposal.proposal_type === "PUBLIC_MESSAGE") {
    return (
      <blockquote className="mt-1 rounded-lg bg-pink-50 px-3 py-2 text-xs italic text-slate-800">
        “{p.message}”
        <div className="mt-1 text-[10px] not-italic text-slate-500">
          {p.channel.replaceAll("_", " ").toLowerCase()} · {p.audience} · {p.language}
        </div>
      </blockquote>
    );
  }
  return <p className="text-[11px] text-slate-600">{fmt(p.quantity)} {p.item_code} for {p.facility_name}</p>;
}

function PendingItem({ proposal, reviewer, onDecided }) {
  const [note, setNote] = useState("");
  const [busy, setBusy] = useState(false);
  const [error, setError] = useState("");

  async function decide(action) {
    setBusy(true);
    setError("");
    try {
      await apiPost(`/api/proposals/${proposal.id}/${action}`, { decided_by: reviewer, note: note || null });
      onDecided();
    } catch (err) {
      setError(err.message);
      setBusy(false);
    }
  }

  return (
    <li className="rounded-xl p-3 ring-1 ring-slate-200">
      <div className="flex items-center justify-between gap-2">
        <Pill className={TYPE_STYLES[proposal.proposal_type]}>{proposal.proposal_type.replaceAll("_", " ")}</Pill>
        <span className="text-[10px] text-slate-400">#{proposal.id} · by {proposal.proposed_by}</span>
      </div>
      <div className="mt-1 text-xs font-semibold text-slate-800">{proposal.title}</div>
      <Details proposal={proposal} />
      {proposal.proposal_type !== "PUBLIC_MESSAGE" && (
        <p className="mt-1 text-[11px] leading-snug text-slate-500">{proposal.summary}</p>
      )}
      <div className="mt-2 flex flex-wrap items-center gap-2">
        <input
          className="min-w-0 flex-1 rounded-lg px-2 py-1 text-[11px] ring-1 ring-slate-200"
          placeholder="Note (optional)"
          value={note}
          onChange={(event) => setNote(event.target.value)}
        />
        <Button onClick={() => decide("approve")} disabled={busy || !reviewer.trim()}>Approve</Button>
        <Button variant="danger" onClick={() => decide("reject")} disabled={busy || !reviewer.trim()}>Reject</Button>
      </div>
      <ErrorNote error={error} />
    </li>
  );
}

export default function ApprovalQueue({ proposals, onChanged }) {
  const [reviewer, setReviewer] = useState(() => localStorage.getItem("nextora-reviewer") || "");
  const pending = proposals.filter((p) => p.status === "PENDING");
  const decided = proposals.filter((p) => p.status !== "PENDING").slice(0, 8);

  function updateReviewer(value) {
    setReviewer(value);
    localStorage.setItem("nextora-reviewer", value);
  }

  return (
    <Card
      title={`Approval queue (${pending.length})`}
      subtitle="Nothing happens until a named person approves. Approving a transfer moves the stock."
    >
      <label className="mb-3 block text-[11px] font-semibold text-slate-500">
        Your name (recorded with each decision)
        <input
          className="mt-1 w-full rounded-lg px-2 py-1.5 text-xs text-slate-800 ring-1 ring-slate-300"
          value={reviewer}
          placeholder="e.g. Dr A. Bello, response coordinator"
          onChange={(event) => updateReviewer(event.target.value)}
        />
      </label>

      <ul className="max-h-[420px] space-y-2 overflow-auto pr-1">
        {pending.map((proposal) => (
          <PendingItem key={proposal.id} proposal={proposal} reviewer={reviewer} onDecided={onChanged} />
        ))}
        {!pending.length && <Empty>No proposals awaiting approval. Run a response to generate some.</Empty>}
      </ul>

      {decided.length > 0 && (
        <div className="mt-4">
          <div className="mb-1 text-[11px] font-semibold text-slate-500">Recent decisions</div>
          <ul className="space-y-1">
            {decided.map((p) => (
              <li key={p.id} className="flex items-start gap-2 text-[11px] text-slate-600">
                <Pill className={STATUS_STYLES[p.status]}>{p.status}</Pill>
                <span className="min-w-0">
                  {p.title} <span className="text-slate-400">· {p.decided_by}</span>
                  {p.execution_result?.error && <span className="block text-red-600">{p.execution_result.error}</span>}
                  {p.execution_result?.note && <span className="block text-slate-400">{p.execution_result.note}</span>}
                </span>
              </li>
            ))}
          </ul>
        </div>
      )}
    </Card>
  );
}
