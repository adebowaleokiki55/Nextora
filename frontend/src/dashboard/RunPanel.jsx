import { useEffect, useRef, useState } from "react";
import ReactMarkdown from "react-markdown";
import remarkGfm from "remark-gfm";
import { apiGet, apiPost, eventsUrl } from "./api";
import { Button, Card, Empty, ErrorNote, Pill } from "./ui";

const AGENT_STYLES = {
  Commander: "bg-slate-900 text-white",
  Surveillance: "bg-sky-100 text-sky-800",
  Supply: "bg-violet-100 text-violet-800",
  Outreach: "bg-pink-100 text-pink-800",
  Policy: "bg-emerald-100 text-emerald-800",
};

const KIND_LABEL = {
  THOUGHT: "thinking",
  TOOL_CALL: "calls tool",
  TOOL_RESULT: "tool result",
  HANDOFF: "hand-off",
  FINAL: "final answer",
  ERROR: "error",
};

function pretty(content) {
  try {
    return JSON.stringify(JSON.parse(content), null, 2);
  } catch {
    return content;
  }
}

function FeedItem({ message }) {
  const agentStyle = AGENT_STYLES[message.agent] || "bg-slate-100 text-slate-700";
  const target = message.payload?.to;
  const isTool = message.kind === "TOOL_CALL" || message.kind === "TOOL_RESULT";

  return (
    <li className={`rounded-lg px-3 py-2 ${message.kind === "ERROR" ? "bg-red-50" : message.kind === "FINAL" ? "bg-emerald-50" : "bg-slate-50"}`}>
      <div className="flex flex-wrap items-center gap-1.5 text-[11px]">
        <Pill className={agentStyle}>{message.agent}</Pill>
        {target && (
          <>
            <span className="text-slate-400">→</span>
            <Pill className={AGENT_STYLES[target] || "bg-slate-100"}>{target}</Pill>
          </>
        )}
        <span className="text-slate-500">{KIND_LABEL[message.kind] || message.kind}</span>
        {message.tool_name && <code className="rounded bg-white px-1 text-[10px] text-slate-700 ring-1 ring-slate-200">{message.tool_name}</code>}
      </div>
      {isTool ? (
        <details className="mt-1">
          <summary className="cursor-pointer text-[11px] text-slate-500">
            {message.kind === "TOOL_CALL" ? "arguments" : "result"} ({message.content.length} chars)
          </summary>
          <pre className="mt-1 max-h-48 overflow-auto whitespace-pre-wrap rounded bg-white p-2 text-[10px] leading-snug text-slate-700 ring-1 ring-slate-200">
            {pretty(message.content)}
          </pre>
        </details>
      ) : (
        <div className="nx-markdown mt-1 text-xs leading-relaxed text-slate-800">
          <ReactMarkdown remarkPlugins={[remarkGfm]}>{message.content}</ReactMarkdown>
        </div>
      )}
    </li>
  );
}

export default function RunPanel({ locations, selectedId, onSelect, health, onFinished }) {
  const [mode, setMode] = useState("auto");
  const [run, setRun] = useState(null);
  const [messages, setMessages] = useState([]);
  const [running, setRunning] = useState(false);
  const [error, setError] = useState("");
  const sourceRef = useRef(null);
  const listRef = useRef(null);

  useEffect(() => () => sourceRef.current?.close(), []);

  useEffect(() => {
    if (listRef.current) listRef.current.scrollTop = listRef.current.scrollHeight;
  }, [messages]);

  async function start() {
    setError("");
    setMessages([]);
    setRun(null);
    setRunning(true);
    sourceRef.current?.close();

    try {
      const started = await apiPost("/api/command/runs", { location_id: selectedId, mode });
      setRun(started);
      const source = new EventSource(eventsUrl(started.id));
      sourceRef.current = source;

      source.addEventListener("message", (event) => {
        const message = JSON.parse(event.data);
        setMessages((current) => (current.some((m) => m.seq === message.seq) ? current : [...current, message]));
      });
      source.addEventListener("done", async () => {
        source.close();
        try {
          setRun(await apiGet(`/api/command/runs/${started.id}`));
        } catch (err) {
          setError(err.message);
        }
        setRunning(false);
        onFinished?.();
      });
      source.onerror = () => {
        // EventSource retries by itself; only give up if the run is gone.
        if (source.readyState === EventSource.CLOSED) {
          setRunning(false);
          setError("Lost the live feed. Refresh to see the final result.");
        }
      };
    } catch (err) {
      setError(err.message);
      setRunning(false);
    }
  }

  const liveAvailable = health?.agent_configured;
  const offline = run?.mode === "OFFLINE_SCRIPTED";

  return (
    <Card
      title="AI command centre"
      subtitle="Commander, Surveillance, Supply, Outreach and Policy agents. They can only propose; you approve."
    >
      <div className="flex flex-wrap items-end gap-2">
        <label className="flex flex-col text-[11px] font-semibold text-slate-500">
          Location
          <select
            className="mt-1 rounded-lg bg-white px-2 py-1.5 text-xs text-slate-800 ring-1 ring-slate-300"
            value={selectedId || ""}
            onChange={(event) => onSelect(Number(event.target.value))}
            disabled={running}
          >
            {locations.map((row) => (
              <option key={row.location.id} value={row.location.id}>
                {row.location.name} ({row.risk_level.replace("_", " ").toLowerCase()})
              </option>
            ))}
          </select>
        </label>
        <label className="flex flex-col text-[11px] font-semibold text-slate-500">
          Mode
          <select
            className="mt-1 rounded-lg bg-white px-2 py-1.5 text-xs text-slate-800 ring-1 ring-slate-300"
            value={mode}
            onChange={(event) => setMode(event.target.value)}
            disabled={running}
          >
            <option value="auto">Auto ({liveAvailable ? "live AI" : "offline script"})</option>
            <option value="live" disabled={!liveAvailable}>Live AI{liveAvailable ? "" : " (no provider set)"}</option>
            <option value="offline">Offline demo (scripted)</option>
          </select>
        </label>
        <Button onClick={start} disabled={running || !selectedId} className="py-2">
          {running ? "Running…" : "Run response"}
        </Button>
      </div>

      {health && (
        <p className="mt-2 text-[11px] text-slate-500">
          AI provider: {liveAvailable ? `${health.agent_runtime} · ${health.agent_model}` : "none configured: runs use the offline script"}
        </p>
      )}

      <ErrorNote error={error} />

      {run && (
        <div className="mt-3 flex flex-wrap items-center gap-2 text-[11px]">
          <Pill className={offline ? "border border-dashed border-amber-500 bg-amber-50 text-amber-800" : "bg-indigo-100 text-indigo-800"}>
            {offline ? "Offline demo mode: scripted, no AI model" : `Live AI${run.model ? ` · ${run.model}` : ""}`}
          </Pill>
          <span className="text-slate-500">Run #{run.id} · {run.location.name} · {run.status.toLowerCase()}</span>
        </div>
      )}

      <ol ref={listRef} className="mt-3 max-h-[460px] space-y-1.5 overflow-auto pr-1">
        {messages.map((message) => <FeedItem key={message.seq} message={message} />)}
        {!messages.length && !running && <Empty>Pick a location and press Run response to watch the agents work.</Empty>}
        {running && <li className="animate-pulse px-3 py-2 text-[11px] text-slate-400">Agents working…</li>}
      </ol>

      {run?.situation_report && !running && (
        <div className="mt-3 rounded-xl bg-white p-3 ring-1 ring-emerald-200">
          <div className="text-xs font-bold text-emerald-800">Situation report (Policy Agent)</div>
          <div className="nx-markdown text-xs leading-relaxed text-slate-800">
            <ReactMarkdown remarkPlugins={[remarkGfm]}>{run.situation_report}</ReactMarkdown>
          </div>
        </div>
      )}
    </Card>
  );
}
