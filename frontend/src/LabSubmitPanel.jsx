import { useEffect, useState } from "react";
import { ClipboardCheck, Send } from "lucide-react";
import "./LabWorkflow.css";

const API_URL = import.meta.env.VITE_API_URL || "http://127.0.0.1:8000";

function todayISO() {
  const now = new Date();
  const offset = now.getTimezoneOffset() * 60000;
  return new Date(now - offset).toISOString().slice(0, 10);
}

/**
 * Saves a whole-smear image as a lab submission that waits for qualified
 * review. The server re-runs the same analysis and stores it as
 * PENDING_REVIEW; nothing feeds surveillance until a reviewer verifies it.
 */
export default function LabSubmitPanel({ file }) {
  const [facilities, setFacilities] = useState([]);
  const [facilityId, setFacilityId] = useState("");
  const [sampleDate, setSampleDate] = useState(todayISO());
  const [sampleReference, setSampleReference] = useState("");
  const [submittedBy, setSubmittedBy] = useState("");
  const [isRealSample, setIsRealSample] = useState(false);
  const [saving, setSaving] = useState(false);
  const [saved, setSaved] = useState(null);
  const [error, setError] = useState("");

  useEffect(() => {
    let cancelled = false;

    fetch(`${API_URL}/api/facilities?has_laboratory=true`)
      .then((response) => response.json())
      .then((data) => {
        if (cancelled) return;
        const labs = data.facilities || [];
        setFacilities(labs);
        if (labs.length) setFacilityId(String(labs[0].id));
      })
      .catch(() => {
        if (!cancelled) setError("Could not load the list of laboratories.");
      });

    return () => {
      cancelled = true;
    };
  }, []);

  async function submit(event) {
    event.preventDefault();
    if (!file || saving) return;

    setSaving(true);
    setError("");

    const form = new FormData();
    form.append("file", file);
    form.append("facility_id", facilityId);
    form.append("sample_date", sampleDate);
    form.append("submitted_by", submittedBy);
    if (sampleReference.trim()) form.append("sample_reference", sampleReference.trim());
    form.append("data_origin", isRealSample ? "REAL" : "SYNTHETIC");

    try {
      const response = await fetch(`${API_URL}/api/lab/submissions`, {
        method: "POST",
        body: form,
      });
      const data = await response.json();

      if (!response.ok) {
        const detail = Array.isArray(data.detail)
          ? data.detail.map((item) => item.msg).join(" ")
          : data.detail;
        throw new Error(detail || "Could not save the submission.");
      }

      setSaved(data);
    } catch (err) {
      setError(
        err.message === "Failed to fetch"
          ? "Cannot connect to the Nextora backend."
          : err.message
      );
    } finally {
      setSaving(false);
    }
  }

  if (saved) {
    return (
      <div className="lab-submit-panel">
        <div className="lab-success">
          <ClipboardCheck size={18} />
          <div>
            Saved as lab submission <strong>#{saved.id}</strong> for{" "}
            {saved.facility.name} ({saved.location.name}). Status:{" "}
            <strong>pending review</strong>. It will not be used for
            surveillance until a qualified laboratory professional verifies it.{" "}
            <a href="/?view=lab-review">Open the review queue</a>
          </div>
        </div>
      </div>
    );
  }

  return (
    <form className="lab-submit-panel" onSubmit={submit}>
      <h3>Submit for laboratory review</h3>
      <p>
        Record this smear against a laboratory and sample date. It is stored as
        pending review; a qualified professional must verify it before it
        counts as a surveillance observation.
      </p>

      <div className="lab-field">
        <label htmlFor="lab-facility">Laboratory</label>
        <select
          id="lab-facility"
          value={facilityId}
          onChange={(event) => setFacilityId(event.target.value)}
          required
        >
          {facilities.map((facility) => (
            <option key={facility.id} value={facility.id}>
              {facility.name} · {facility.location_name}
              {facility.data_origin === "SYNTHETIC" ? " (demo facility)" : ""}
            </option>
          ))}
        </select>
      </div>

      <div className="lab-field-row">
        <div className="lab-field">
          <label htmlFor="lab-date">Sample date</label>
          <input
            id="lab-date"
            type="date"
            value={sampleDate}
            max={todayISO()}
            onChange={(event) => setSampleDate(event.target.value)}
            required
          />
        </div>

        <div className="lab-field">
          <label htmlFor="lab-ref">Slide / sample number</label>
          <input
            id="lab-ref"
            value={sampleReference}
            maxLength={60}
            onChange={(event) => setSampleReference(event.target.value)}
            placeholder="Optional, e.g. SL-0042"
          />
        </div>
      </div>
      <div className="lab-field">
        <small>Do not enter patient names or other personal information.</small>
      </div>

      <div className="lab-field">
        <label htmlFor="lab-by">Submitted by</label>
        <input
          id="lab-by"
          value={submittedBy}
          maxLength={120}
          onChange={(event) => setSubmittedBy(event.target.value)}
          placeholder="Your name or role"
          required
        />
      </div>

      <label className="lab-origin-toggle">
        <input
          type="checkbox"
          checked={isRealSample}
          onChange={(event) => setIsRealSample(event.target.checked)}
        />
        <span>
          This is a real patient sample from this laboratory. Leave unticked for
          demo or test images: they are labelled <strong>SYNTHETIC</strong>{" "}
          everywhere they appear.
        </span>
      </label>

      <button
        type="submit"
        className="primary-button"
        disabled={saving || !file || !facilityId || !submittedBy.trim()}
      >
        <Send size={15} />
        {saving ? "Saving…" : "Submit for review"}
      </button>

      {error && <div className="lab-error">{error}</div>}
    </form>
  );
}
