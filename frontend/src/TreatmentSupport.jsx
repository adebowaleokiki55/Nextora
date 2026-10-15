
import { useState } from "react";
import {
  Activity,
  AlertTriangle,
  CheckCircle2,
  ClipboardList,
  LoaderCircle,
  ShieldAlert,
  Sparkles,
} from "lucide-react";
import ReactMarkdown from "react-markdown";
import remarkGfm from "remark-gfm";
import "./TreatmentSupport.css";

const API_URL =
  import.meta.env.VITE_API_URL || "http://127.0.0.1:8000";

const INITIAL_FORM = {
  laboratory_confirmed: false,
  confirmation_method: "not_confirmed",
  malaria_species: "unknown",
  severity: "unknown",
  patient_weight_kg: "",
  age_years: "",
  pregnant: "unknown",
  pregnancy_trimester: "not_applicable",
  known_allergies: "",
  relevant_medications: "",
  parasite_density_per_microlitre: "",
};

function splitList(value) {
  return value
    .split(",")
    .map((item) => item.trim())
    .filter(Boolean);
}

function buildPayload(form) {
  return {
    laboratory_confirmed: form.laboratory_confirmed,
    confirmation_method: form.confirmation_method,
    malaria_species: form.malaria_species,
    severity: form.severity,
    patient_weight_kg:
      form.patient_weight_kg === ""
        ? null
        : Number(form.patient_weight_kg),
    age_years:
      form.age_years === "" ? null : Number(form.age_years),
    pregnant:
      form.pregnant === "unknown"
        ? null
        : form.pregnant === "yes",
    pregnancy_trimester: form.pregnancy_trimester,
    known_allergies: splitList(form.known_allergies),
    relevant_medications: splitList(form.relevant_medications),
    parasite_density_per_microlitre:
      form.parasite_density_per_microlitre === ""
        ? null
        : Number(form.parasite_density_per_microlitre),
  };
}

function humanizeStatus(value = "") {
  return String(value)
    .replaceAll("_", " ")
    .replace(/\b\w/g, (character) => character.toUpperCase());
}

function normalizeAgentMarkdown(value = "") {
  return String(value)
    .replace(/\\<br\s*\/?>/gi, "\n")
    .replace(/&lt;br\s*\/?&gt;/gi, "\n")
    .replace(/\\([*_|`#~])/g, "$1")
    .replace(/\r\n/g, "\n")
    .trim();
}

function MarkdownExplanation({ content }) {
  const markdown = normalizeAgentMarkdown(content);

  return (
    <div className="ts-agent-markdown">
      <ReactMarkdown
        remarkPlugins={[remarkGfm]}
        components={{
          a: ({ children, ...props }) => (
            <a {...props} target="_blank" rel="noreferrer">
              {children}
            </a>
          ),
        }}
      >
        {markdown || "No explanation was returned."}
      </ReactMarkdown>
    </div>
  );
}

export default function TreatmentSupport() {
  const [form, setForm] = useState(INITIAL_FORM);
  const [assessment, setAssessment] = useState(null);
  const [agentReport, setAgentReport] = useState(null);
  const [loadingAssessment, setLoadingAssessment] = useState(false);
  const [loadingAgent, setLoadingAgent] = useState(false);
  const [error, setError] = useState("");

  const updateField = (name, value) => {
    setForm((current) => ({
      ...current,
      [name]: value,
    }));
    setError("");
  };

  async function requestAssessment() {
    setLoadingAssessment(true);
    setError("");
    setAssessment(null);
    setAgentReport(null);

    try {
      const response = await fetch(
        `${API_URL}/api/treatment/recommend`,
        {
          method: "POST",
          headers: {
            "Content-Type": "application/json",
          },
          body: JSON.stringify(buildPayload(form)),
        }
      );

      const data = await response.json().catch(() => ({}));

      if (!response.ok) {
        throw new Error(
          data.detail || "Treatment assessment could not be completed."
        );
      }

      setAssessment(data);
    } catch (err) {
      setError(
        err.message ||
          "Unable to connect to the treatment assessment service."
      );
    } finally {
      setLoadingAssessment(false);
    }
  }

  async function requestAgentReport() {
    if (!assessment) {
      setError(
        "Run the treatment assessment before requesting an AI explanation."
      );
      return;
    }

    setLoadingAgent(true);
    setError("");
    setAgentReport(null);

    try {
      const response = await fetch(
        `${API_URL}/api/agent/treatment-report`,
        {
          method: "POST",
          headers: {
            "Content-Type": "application/json",
          },
          body: JSON.stringify(buildPayload(form)),
        }
      );

      const data = await response.json().catch(() => ({}));

      if (!response.ok) {
        throw new Error(
          data.detail || "The AI explanation could not be generated."
        );
      }

      if (!data.explanation) {
        throw new Error(
          "The AI service returned no explanation. Please try again."
        );
      }

      setAgentReport(data);
    } catch (err) {
      setError(
        err.message ||
          "Unable to connect to the AI explanation service."
      );
    } finally {
      setLoadingAgent(false);
    }
  }

  const isEmergency =
    assessment?.status === "urgent_escalation" ||
    assessment?.urgency === "emergency";

  return (
    <section className="treatment-support">
      <header className="ts-header">
        <div className="ts-icon">
          <Activity size={21} />
        </div>

        <div>
          <p className="ts-eyebrow">CLINICAL WORKFLOW</p>
          <h2>Treatment decision support</h2>
          <p className="ts-subtitle">
            Preliminary protocol-aware assessment for clinician review.
          </p>
        </div>
      </header>

      <div className="ts-safety-note">
        <ShieldAlert size={19} />
        <p>
          Nextora's image prediction does not confirm malaria. Enter
          clinical information from an appropriate assessment and
          laboratory report. This prototype is not clinically validated
          and does not prescribe medication.
        </p>
      </div>

      <div className="ts-form">
        <div className="ts-field ts-checkbox-field">
          <label className="ts-checkbox-label">
            <input
              type="checkbox"
              checked={form.laboratory_confirmed}
              onChange={(event) =>
                updateField(
                  "laboratory_confirmed",
                  event.target.checked
                )
              }
            />
            <span>
              Malaria confirmation has been reported by a clinician
              or laboratory
            </span>
          </label>

          <small>
            This checkbox is user-reported; Nextora does not independently
            verify a laboratory result.
          </small>
        </div>

        <div className="ts-grid">
          <div className="ts-field">
            <label htmlFor="ts-confirmation">Confirmation method</label>
            <select
              id="ts-confirmation"
              value={form.confirmation_method}
              onChange={(event) =>
                updateField("confirmation_method", event.target.value)
              }
            >
              <option value="not_confirmed">Not confirmed</option>
              <option value="microscopy">Microscopy</option>
              <option value="rapid_diagnostic_test">
                Rapid diagnostic test (RDT)
              </option>
              <option value="other">Other</option>
            </select>
          </div>

          <div className="ts-field">
            <label htmlFor="ts-species">Malaria species</label>
            <select
              id="ts-species"
              value={form.malaria_species}
              onChange={(event) =>
                updateField("malaria_species", event.target.value)
              }
            >
              <option value="unknown">Unknown / not established</option>
              <option value="falciparum">P. falciparum</option>
              <option value="vivax">P. vivax</option>
              <option value="ovale">P. ovale</option>
              <option value="malariae">P. malariae</option>
              <option value="knowlesi">P. knowlesi</option>
            </select>
          </div>

          <div className="ts-field">
            <label htmlFor="ts-severity">Clinical severity</label>
            <select
              id="ts-severity"
              value={form.severity}
              onChange={(event) =>
                updateField("severity", event.target.value)
              }
            >
              <option value="unknown">Not assessed / unknown</option>
              <option value="uncomplicated">Uncomplicated</option>
              <option value="severe">Severe / suspected severe</option>
            </select>
          </div>

          <div className="ts-field">
            <label htmlFor="ts-weight">Weight (kg)</label>
            <input
              id="ts-weight"
              type="number"
              min="0.1"
              max="300"
              step="0.1"
              placeholder="e.g. 60"
              value={form.patient_weight_kg}
              onChange={(event) =>
                updateField("patient_weight_kg", event.target.value)
              }
            />
          </div>

          <div className="ts-field">
            <label htmlFor="ts-age">Age (years)</label>
            <input
              id="ts-age"
              type="number"
              min="0"
              max="120"
              step="0.1"
              placeholder="e.g. 25"
              value={form.age_years}
              onChange={(event) =>
                updateField("age_years", event.target.value)
              }
            />
          </div>

          <div className="ts-field">
            <label htmlFor="ts-pregnant">Pregnancy status</label>
            <select
              id="ts-pregnant"
              value={form.pregnant}
              onChange={(event) => {
                const value = event.target.value;

                setForm((current) => ({
                  ...current,
                  pregnant: value,
                  pregnancy_trimester:
                    value === "yes"
                      ? current.pregnancy_trimester === "not_applicable"
                        ? "unknown"
                        : current.pregnancy_trimester
                      : "not_applicable",
                }));

                setError("");
              }}
            >
              <option value="unknown">Unknown / not assessed</option>
              <option value="yes">Pregnant</option>
              <option value="no">Not pregnant</option>
            </select>
          </div>

          {form.pregnant === "yes" && (
            <div className="ts-field">
              <label htmlFor="ts-trimester">Pregnancy trimester</label>
              <select
                id="ts-trimester"
                value={form.pregnancy_trimester}
                onChange={(event) =>
                  updateField("pregnancy_trimester", event.target.value)
                }
              >
                <option value="unknown">Unknown</option>
                <option value="first">First trimester</option>
                <option value="second">Second trimester</option>
                <option value="third">Third trimester</option>
              </select>
            </div>
          )}

          <div className="ts-field">
            <label htmlFor="ts-density">
              Lab-reported parasite density / µL
            </label>
            <input
              id="ts-density"
              type="number"
              min="0"
              step="any"
              placeholder="Optional"
              value={form.parasite_density_per_microlitre}
              onChange={(event) =>
                updateField(
                  "parasite_density_per_microlitre",
                  event.target.value
                )
              }
            />
            <small>
              Do not estimate this from Nextora's candidate counts.
            </small>
          </div>

          <div className="ts-field ts-full">
            <label htmlFor="ts-allergies">Known allergies</label>
            <input
              id="ts-allergies"
              type="text"
              placeholder="Separate multiple entries with commas"
              value={form.known_allergies}
              onChange={(event) =>
                updateField("known_allergies", event.target.value)
              }
            />
          </div>

          <div className="ts-field ts-full">
            <label htmlFor="ts-medications">Current medications</label>
            <input
              id="ts-medications"
              type="text"
              placeholder="Separate multiple entries with commas"
              value={form.relevant_medications}
              onChange={(event) =>
                updateField("relevant_medications", event.target.value)
              }
            />
          </div>
        </div>

        <button
          className="ts-primary-button"
          type="button"
          onClick={requestAssessment}
          disabled={loadingAssessment || loadingAgent}
        >
          {loadingAssessment ? (
            <LoaderCircle className="ts-spin" size={18} />
          ) : (
            <ClipboardList size={18} />
          )}
          {loadingAssessment
            ? "Assessing..."
            : "Assess treatment pathway"}
        </button>
      </div>

      {error && (
        <div className="ts-error" role="alert">
          <AlertTriangle size={18} />
          <span>{error}</span>
        </div>
      )}

      {assessment && (
        <div className="ts-results" aria-live="polite">
          <div className="ts-result-heading">
            <div>
              <p className="ts-eyebrow">ASSESSMENT RESULT</p>
              <h3>{humanizeStatus(assessment.status)}</h3>
            </div>

            <span
              className={`ts-status ${
                isEmergency ? "ts-status-danger" : ""
              }`}
            >
              {humanizeStatus(assessment.urgency)}
            </span>
          </div>

          {isEmergency && (
            <div className="ts-emergency">
              <AlertTriangle size={20} />
              <div>
                <strong>Emergency assessment required</strong>
                <p>
                  Do not delay urgent hospital care to use this application.
                </p>
              </div>
            </div>
          )}

          {assessment.recommendation && (
            <div className="ts-recommendation">
              <h4>Recommendation</h4>
              <p>{assessment.recommendation}</p>
            </div>
          )}

          {assessment.medication && (
            <div className="ts-medication">
              <h4>Medication pathway under review</h4>

              {assessment.medication.class && (
                <p>
                  <strong>Class:</strong> {assessment.medication.class}
                </p>
              )}

              {assessment.medication.candidate_regimen && (
                <p>
                  <strong>Candidate regimen:</strong>{" "}
                  {assessment.medication.candidate_regimen}
                </p>
              )}

              {assessment.medication.selection_status && (
                <p>
                  <strong>Status:</strong>{" "}
                  {assessment.medication.selection_status}
                </p>
              )}
            </div>
          )}

          <div className="ts-dose">
            <span>Dose</span>
            <strong>
              {assessment.dose == null
                ? "Not calculated"
                : String(assessment.dose)}
            </strong>
            <small>
              This prototype does not provide a prescription.
            </small>
          </div>

          {assessment.missing_information?.length > 0 && (
            <div className="ts-list-section">
              <h4>Information still needed</h4>
              <ul>
                {assessment.missing_information.map((item, index) => (
                  <li key={`${item}-${index}`}>{item}</li>
                ))}
              </ul>
            </div>
          )}

          {assessment.reasons?.length > 0 && (
            <div className="ts-list-section">
              <h4>Assessment notes</h4>
              <ul>
                {assessment.reasons.map((item, index) => (
                  <li key={`${item}-${index}`}>{item}</li>
                ))}
              </ul>
            </div>
          )}

          {assessment.warnings?.length > 0 && (
            <div className="ts-warnings">
              <h4>
                <AlertTriangle size={17} />
                Important warnings
              </h4>
              <ul>
                {assessment.warnings.map((item, index) => (
                  <li key={`${item}-${index}`}>{item}</li>
                ))}
              </ul>
            </div>
          )}

          {assessment.protocol && (
            <div className="ts-protocol">
              <h4>Protocol reference</h4>
              <p>
                {assessment.protocol.name} —{" "}
                {assessment.protocol.publisher},{" "}
                {assessment.protocol.year}
              </p>

              {assessment.protocol.dosing_table_verified === false && (
                <p className="ts-unverified">
                  Dosing table not verified in this application.
                </p>
              )}

              {assessment.protocol.url && (
                <a
                  href={assessment.protocol.url}
                  target="_blank"
                  rel="noreferrer"
                >
                  Open protocol reference ↗
                </a>
              )}
            </div>
          )}

          <button
            className="ts-secondary-button"
            type="button"
            onClick={requestAgentReport}
            disabled={loadingAgent || loadingAssessment}
          >
            {loadingAgent ? (
              <LoaderCircle className="ts-spin" size={18} />
            ) : (
              <Sparkles size={18} />
            )}
            {loadingAgent
              ? "Generating explanation..."
              : "Explain with Nextora AI"}
          </button>

          {agentReport && (
            <div className="ts-agent-report">
              <h4>
                <Sparkles size={17} />
                AI explanation
              </h4>

              <MarkdownExplanation content={agentReport.explanation} />

              <div className="ts-agent-meta">
                <CheckCircle2 size={15} />
                Deterministic assessment tool executed
                {agentReport.agent_model
                  ? ` · ${agentReport.agent_model}`
                  : ""}
              </div>

              {agentReport.disclaimer && (
                <p className="ts-agent-disclaimer">
                  {agentReport.disclaimer}
                </p>
              )}
            </div>
          )}

          <p className="ts-footer-disclaimer">
            Research prototype only. A qualified clinician must assess the
            patient, confirm relevant findings, and verify current applicable
            treatment guidance. This tool does not diagnose or prescribe.
          </p>
        </div>
      )}
    </section>
  );
}
