
import LandingPage from "./LandingPage";
import Analyzer from "./Analyzer";
import TreatmentSupport from "./TreatmentSupport";
import { ArrowLeft, Microscope } from "lucide-react";
import "./LandingPage.css";

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
