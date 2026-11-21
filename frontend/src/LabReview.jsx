import { useCallback, useEffect, useState } from "react";
import { AlertCircle, ClipboardList } from "lucide-react";
import "./LabWorkflow.css";

const API_URL = import.meta.env.VITE_API_URL || "http://127.0.0.1:8000";

const TABS = [
  { key: "PENDING_REVIEW", label: "Pending review" },
  { key: "VERIFIED", label: "Verified" },
  { key: "REJECTED", label: "Rejected" },
  { key: "", label: "All" },
];

const STATUS_LABEL = {
  PENDING_REVIEW: "Pending review",
  VERIFIED: "Verified",
  REJECTED: "Rejected",
};

function StatusBadge({ status }) {
  const style = {
    PENDING_REVIEW: "pending",
    VERIFIED: "verified",
    REJECTED: "rejected",
  }[status];
  return <span className={`lab-badge ${style}`}>{STATUS_LABEL[status] || status}</span>;
}

export function OriginBadge({ origin }) {
  return origin === "SYNTHETIC" ? (
    <span className="lab-badge synthetic" title="Demo or test data, not a real observation">
      Synthetic
    </span>
  ) : (
    <span className="lab-badge real">Real</span>
  );
}

async function readJson(response) {
  const data = await response.json();
  if (!response.ok) {
    const detail = Array.isArray(data.detail)
      ? data.detail.map((item) => item.msg).join(" ")
      : data.detail;
    throw new Error(detail || "Request failed.");
  }
  return data;
}

function ReviewForm({ submission, onReviewed }) {
  const [reviewerName, setReviewerName] = useState("");
  const [notes, setNotes] = useState("");
  const [busy, setBusy] = useState(false);
  const [error, setError] = useState("");

  async function send(decision, reviewerResult) {
    if (busy) return;
    setBusy(true);
    setError("");

    try {
      const data = await readJson(
        await fetch(`${API_URL}/api/lab/submissions/${submission.id}/review`, {
          method: "POST",
          headers: { "Content-Type": "application/json" },
          body: JSON.stringify({
            decision,
            reviewer_name: reviewerName,
            reviewer_result: reviewerResult,
            notes: notes.trim() || null,
          }),
        })
      );
      onReviewed(data);
    } catch (err) {
      setError(err.message);
    } finally {
      setBusy(false);
    }
  }

  const ready = reviewerName.trim().length > 0 && !busy;

  return (
    <div className="lab-review-form">
      <h3>Qualified review</h3>
      <p>
        Examine the slide yourself. Your own reading is what gets recorded,
        not the model&apos;s. Verified results become surveillance observations
        for {submission.location.name}; a verified positive also raises an
        alert in the command centre.
      </p>

      <div className="lab-field">
        <label htmlFor="reviewer">Reviewer name</label>
        <input
          id="reviewer"
          value={reviewerName}
          maxLength={120}
          onChange={(event) => setReviewerName(event.target.value)}
          placeholder="Qualified laboratory professional"
        />
      </div>

      <div className="lab-field">
        <label htmlFor="notes">Notes</label>
        <textarea
          id="notes"
          value={notes}
          maxLength={2000}
          onChange={(event) => setNotes(event.target.value)}
          placeholder="Required when rejecting (e.g. out of focus, poor stain, inconclusive)"
        />
      </div>

      <div className="lab-decisions">
        <button
          type="button"
          className="positive"
          disabled={!ready}
          onClick={() => send("VERIFIED", "POSITIVE")}
        >
          Verify: positive
        </button>
        <button
          type="button"
          className="negative"
          disabled={!ready}
          onClick={() => send("VERIFIED", "NEGATIVE")}
        >
          Verify: negative
        </button>
        <button
          type="button"
          className="reject"
          disabled={!ready || !notes.trim()}
          onClick={() => send("REJECTED", null)}
        >
          Reject
        </button>
      </div>

      {error && <div className="lab-error">{error}</div>}
    </div>
  );
}

function SubmissionDetail({ submissionId, onReviewed }) {
  const [submission, setSubmission] = useState(null);
  const [error, setError] = useState("");

  // Remounted per submission (see `key` below), so state starts empty.
  useEffect(() => {
    let cancelled = false;

    fetch(`${API_URL}/api/lab/submissions/${submissionId}`)
      .then(readJson)
      .then((data) => !cancelled && setSubmission(data))
      .catch((err) => !cancelled && setError(err.message));

    return () => {
      cancelled = true;
    };
  }, [submissionId]);

  if (error) return <div className="lab-card lab-error">{error}</div>;
  if (!submission) return <div className="lab-card lab-empty">Loading…</div>;

  const flagged = submission.cells.filter((cell) => cell.predicted_class === 0);

  return (
    <div className="lab-card">
      <div className="lab-detail-top">
        <div className="lab-image-frame">
          <img
            src={`${API_URL}/api/lab/submissions/${submission.id}/annotated.png`}
            alt={`Annotated smear for submission ${submission.id}`}
          />
        </div>

        <div>
          <div className="lab-facts">
            <div className="fact">
              <span>Submission</span>
              <strong>
                #{submission.id} <StatusBadge status={submission.status} />{" "}
                <OriginBadge origin={submission.data_origin} />
              </strong>
            </div>
            <div className="fact"><span>Laboratory</span><strong>{submission.facility.name}</strong></div>
            <div className="fact"><span>Location</span><strong>{submission.location.name}</strong></div>
            <div className="fact"><span>Sample date</span><strong>{submission.sample_date}</strong></div>
            <div className="fact"><span>Slide / sample no.</span><strong>{submission.sample_reference || "—"}</strong></div>
            <div className="fact"><span>Submitted by</span><strong>{submission.submitted_by}</strong></div>
            <div className="fact"><span>Model</span><strong>{submission.model_name} ({submission.model_version})</strong></div>
            {submission.reviewed_by && (
              <>
                <div className="fact"><span>Reviewed by</span><strong>{submission.reviewed_by}</strong></div>
                <div className="fact">
                  <span>Reviewer&apos;s reading</span>
                  <strong>{submission.reviewer_result || "— (rejected)"}</strong>
                </div>
                {submission.review_notes && (
                  <div className="fact"><span>Notes</span><strong>{submission.review_notes}</strong></div>
                )}
              </>
            )}
          </div>

          <div className="lab-counts">
            <div><span>Candidate regions</span><strong>{submission.candidate_count}</strong></div>
            <div><span>Model: possibly parasitized</span><strong>{submission.candidate_infected_count}</strong></div>
            <div><span>Model: uninfected</span><strong>{submission.candidate_uninfected_count}</strong></div>
          </div>

          <div className="disclaimer">
            <AlertCircle size={16} />
            <p>
              Model output is decision support only: candidate regions may not
              be individual cells, scores are not calibrated probabilities, and
              counts are not a validated parasite count.
            </p>
          </div>
        </div>
      </div>

      {flagged.length > 0 && (
        <table className="lab-cells-table">
          <thead>
            <tr><th>Region</th><th>Model class</th><th>Model score</th><th>Position (x, y)</th></tr>
          </thead>
          <tbody>
            {flagged.slice(0, 12).map((cell) => (
              <tr key={cell.cell_id}>
                <td>#{cell.cell_id}</td>
                <td>Possibly parasitized</td>
                <td>{Number(cell.confidence).toFixed(1)}%</td>
                <td>{cell.box.x}, {cell.box.y}</td>
              </tr>
            ))}
          </tbody>
        </table>
      )}

      {submission.status === "PENDING_REVIEW" && (
        <ReviewForm
          submission={submission}
          onReviewed={(updated) => {
            setSubmission(updated);
            onReviewed(updated);
          }}
        />
      )}
    </div>
  );
}

function ModelCard() {
  const [card, setCard] = useState(null);

  useEffect(() => {
    fetch(`${API_URL}/api/lab/model-card`)
      .then(readJson)
      .then(setCard)
      .catch(() => setCard(null));
  }, []);

  if (!card) return null;

  return (
    <div className="lab-card lab-model-card">
      <h3>About the model</h3>
      <div>
        <strong>{card.model_name}</strong> · {card.model_version}
      </div>
      <div>Trained on: {card.training_data}</div>
      <div>{card.intended_use}</div>
      <div style={{ marginTop: 6 }}>
        <strong>Performance: </strong>
        {card.metrics
          ? `accuracy ${card.metrics.accuracy}, sensitivity ${card.metrics.sensitivity}, specificity ${card.metrics.specificity}, AUC ${card.metrics.auc}. ${card.metrics_provenance}`
          : card.metrics_provenance}
      </div>
      <ul>
        {card.limitations.map((item) => <li key={item}>{item}</li>)}
      </ul>
    </div>
  );
}

export default function LabReview() {
  const [tab, setTab] = useState("PENDING_REVIEW");
  const [submissions, setSubmissions] = useState([]);
  const [selectedId, setSelectedId] = useState(null);
  const [error, setError] = useState("");

  const load = useCallback(() => {
    const query = tab ? `?status=${tab}` : "";
    fetch(`${API_URL}/api/lab/submissions${query}`)
      .then(readJson)
      .then((data) => {
        setSubmissions(data.submissions);
        setError("");
      })
      .catch((err) =>
        setError(
          err.message === "Failed to fetch"
            ? "Cannot connect to the Nextora backend."
            : err.message
        )
      );
  }, [tab]);

  useEffect(load, [load]);

  return (
    <main className="lab-review-main">
      <div className="lab-review-head">
        <div>
          <span className="step-label">MODULE 5 · LABORATORY INTELLIGENCE</span>
          <h1>
            <ClipboardList size={24} style={{ verticalAlign: "-3px" }} /> Lab review queue
          </h1>
          <p>
            AI results wait here until a qualified laboratory professional
            verifies or rejects them. Only verified results feed local malaria
            surveillance.
          </p>
        </div>

        <div className="lab-tabs" role="tablist">
          {TABS.map((item) => (
            <button
              key={item.key || "all"}
              type="button"
              role="tab"
              aria-selected={tab === item.key}
              className={tab === item.key ? "active" : ""}
              onClick={() => {
                setTab(item.key);
                setSelectedId(null);
              }}
            >
              {item.label}
            </button>
          ))}
        </div>
      </div>

      {error && <div className="lab-card lab-error">{error}</div>}

      <div className="lab-review-grid">
        <div className="lab-card">
          <div className="lab-queue">
            {submissions.length === 0 ? (
              <div className="lab-empty">
                Nothing here. Upload a whole-smear image in the{" "}
                <a href="/?view=analyzer">Cell Explorer</a> and choose
                &quot;Submit for review&quot;.
              </div>
            ) : (
              submissions.map((item) => (
                <button
                  key={item.id}
                  type="button"
                  className={`lab-queue-item ${item.id === selectedId ? "selected" : ""}`}
                  onClick={() => setSelectedId(item.id)}
                >
                  <div className="line">
                    <strong>#{item.id} · {item.facility.name}</strong>
                    <OriginBadge origin={item.data_origin} />
                  </div>
                  <div className="line">
                    <span>{item.location.name} · {item.sample_date}</span>
                    <StatusBadge status={item.status} />
                  </div>
                  <span>
                    Model flagged {item.candidate_infected_count} of{" "}
                    {item.candidate_count} regions
                  </span>
                </button>
              ))
            )}
          </div>
        </div>

        <div>
          {selectedId ? (
            <SubmissionDetail key={selectedId} submissionId={selectedId} onReviewed={load} />
          ) : (
            <div className="lab-card lab-empty">Select a submission to review it.</div>
          )}
          <ModelCard />
        </div>
      </div>
    </main>
  );
}
