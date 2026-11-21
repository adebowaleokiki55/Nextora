
import { Suspense, lazy } from "react";
import LandingPage from "./LandingPage";
import Analyzer from "./Analyzer";
import TreatmentSupport from "./TreatmentSupport";
import LabReview from "./LabReview";
import { ArrowLeft, Microscope } from "lucide-react";
import "./LandingPage.css";

// Loaded on its own so Tailwind and the map/chart libraries stay off other pages.
const Dashboard = lazy(() => import("./dashboard/Dashboard"));

function WorkspaceHeader({ title }) {
  return (
    <header className="nx-workspace-header">
      <a className="nx-brand" href="/" aria-label="Back to Nextora home">
        <span className="nx-brand-mark">
          <Microscope size={21} />
        </span>
        <span className="nx-brand-copy">
          <strong>nextora<span>.</span></strong>
          <small>AI MICROSCOPY</small>
        </span>
      </a>

      <a className="nx-back-home" href="/">
        <ArrowLeft size={16} />
        Back to home
      </a>

      <span className="nx-workspace-title">{title}</span>
    </header>
  );
}

export default function App() {
  const view = new URLSearchParams(window.location.search).get("view");

  if (view === "analyzer") {
    return <Analyzer />;
  }

  if (view === "command") {
    return (
      <Suspense fallback={<div style={{ padding: 40, fontFamily: "sans-serif" }}>Loading command centre…</div>}>
        <Dashboard />
      </Suspense>
    );
  }

  if (view === "lab-review") {
    return (
      <div className="nx-landing">
        <WorkspaceHeader title="Lab review queue" />
        <LabReview />
        <footer className="nx-workspace-disclaimer">
          Decision-support research prototype · Every AI result requires
          qualified laboratory review · Not a diagnostic tool
        </footer>
      </div>
    );
  }

  if (view === "treatment") {
    return (
      <div className="nx-landing nx-treatment-page">
        <WorkspaceHeader title="Treatment support" />
        <main className="nx-treatment-main">
          <TreatmentSupport />
        </main>
        <footer className="nx-workspace-disclaimer">
          Research prototype only · Not for clinical diagnosis or prescribing
        </footer>
      </div>
    );
  }

  return <LandingPage />;
}
