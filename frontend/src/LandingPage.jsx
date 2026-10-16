
import {
  ArrowDown,
  ArrowRight,
  Activity,
  BrainCircuit,
  Check,
  ChevronRight,
  Microscope,
  ShieldCheck,
  Sparkles,
  Upload,
  ScanSearch,
  FlaskConical,
} from "lucide-react";
import "./LandingPage.css";

const features = [
  {
    number: "01",
    icon: Microscope,
    title: "Single-cell analysis",
    description:
      "Explore an isolated blood-cell image with the PlasmoSENet classification model and inspect its output scores.",
    tag: "IMAGE CLASSIFICATION",
  },
  {
    number: "02",
    icon: ScanSearch,
    title: "Whole-smear exploration",
    description:
      "Examine experimental candidate regions across a microscopy image and review their individual model classifications.",
    tag: "EXPERIMENTAL COMPUTER VISION",
  },
  {
    number: "03",
    icon: BrainCircuit,
    title: "AI-assisted explanations",
    description:
      "Explore readable explanations and a protocol-aware treatment-support workflow intended for further professional review.",
    tag: "AI SUPPORT",
  },
];

const steps = [
  {
    number: "01",
    title: "Upload a sample",
    description:
      "Choose a supported microscopy image from your device.",
  },
  {
    number: "02",
    title: "Run the analysis",
    description:
      "Select Single Cell or Whole Smear and submit the image for processing.",
  },
  {
    number: "03",
    title: "Review the output",
    description:
      "Inspect the model results, candidate regions where available, and the limitations of the analysis.",
  },
];

function MicroscopyVisual() {
  return (
    <div className="nx-visual" aria-label="Abstract microscopy illustration">
      <div className="nx-visual-top">
        <span className="nx-live-dot" />
        <span>MICROSCOPY LAB</span>
        <span className="nx-visual-id">NX / 001</span>
      </div>

      <div className="nx-microscope-field">
        <div className="nx-cell nx-cell-one">
          <span />
        </div>
        <div className="nx-cell nx-cell-two">
          <span />
        </div>
        <div className="nx-cell nx-cell-three">
          <span />
        </div>
        <div className="nx-cell nx-cell-four">
          <span />
        </div>
        <div className="nx-cell nx-cell-five">
          <span />
        </div>
        <div className="nx-cell nx-cell-six">
          <span />
        </div>

        <div className="nx-scan-line" />
        <div className="nx-crosshair nx-crosshair-one" />
        <div className="nx-crosshair nx-crosshair-two" />

        <div className="nx-field-label">
          <span>IMAGE FIELD</span>
          <strong>CV / 01</strong>
        </div>
      </div>

      <div className="nx-visual-bottom">
        <div>
          <span>COMPUTER VISION</span>
          <strong>Microscopy exploration</strong>
        </div>
        <div className="nx-visual-symbol">
          <Activity size={19} />
        </div>
      </div>
    </div>
  );
}

export default function LandingPage() {
  const openAnalyzer = () => {
    window.location.href = "/?view=analyzer";
  };

  const openTreatment = () => {
    window.location.href = "/?view=treatment";
  };

  return (
    <div className="nx-landing">
      <header className="nx-navbar">
        <a className="nx-brand" href="/" aria-label="Nextora homepage">
          <span className="nx-brand-mark">
            <Microscope size={22} strokeWidth={2.1} />
          </span>
          <span className="nx-brand-copy">
            <strong>nextora<span>.</span></strong>
            <small>AI MICROSCOPY</small>
          </span>
        </a>

        <nav className="nx-nav-links" aria-label="Main navigation">
          <a href="#platform">Platform</a>
          <a href="#workflow">How it works</a>
          <a href="#research">Research notes</a>
        </nav>

        <button className="nx-nav-cta" onClick={openAnalyzer}>
          Open analyzer <ArrowRight size={16} />
        </button>
      </header>

      <main>
        <section className="nx-hero">
          <div className="nx-hero-copy">
            <div className="nx-eyebrow">
              <span className="nx-eyebrow-dot" />
              COMPUTER VISION · HEALTH RESEARCH
            </div>

            <h1>
              A closer look.
              <br />
              <span>A new perspective.</span>
            </h1>

            <p className="nx-hero-description">
              An experimental AI microscopy platform exploring how
              computer vision can help analyze blood-cell imagery and
              make model outputs easier to understand.
            </p>

            <div className="nx-hero-actions">
              <button className="nx-button-primary" onClick={openAnalyzer}>
                Try the analyzer <ArrowRight size={18} />
              </button>
              <a className="nx-button-secondary" href="#platform">
                Explore the platform <ArrowDown size={16} />
              </a>
            </div>

            <div className="nx-hero-trust">
              <span><Check size={15} /> Computer-vision research</span>
              <span><Check size={15} /> Transparent limitations</span>
            </div>
          </div>

          <div className="nx-hero-art">
            <div className="nx-art-glow" />
            <MicroscopyVisual />
            <div className="nx-floating-note">
              <span className="nx-note-icon">
                <BrainCircuit size={18} />
              </span>
              <span>
                <strong>AI-assisted analysis</strong>
                <small>Experimental research prototype</small>
              </span>
            </div>
            <div className="nx-art-index">01 — 03 / EXPLORE</div>
          </div>

          <div className="nx-hero-index">
            <span>01</span>
            <span />
            <span>INTRODUCTION</span>
          </div>
        </section>

        <section className="nx-principles">
          <div>
            <span className="nx-section-kicker">THE IDEA</span>
            <h2>
              Built to explore.
              <br />
              Designed to explain.
            </h2>
          </div>
          <p>
            Nextora brings image analysis and AI-assisted explanations
            into one research workspace. The aim is to make experimental
            computer-vision outputs easier to inspect—not to replace
            laboratory testing or clinical judgment.
          </p>
        </section>

        <section className="nx-platform" id="platform">
          <div className="nx-section-heading">
            <div>
              <span className="nx-section-kicker">THE PLATFORM</span>
              <h2>Three parts. One workspace.</h2>
            </div>
            <p>
              Explore the capabilities currently available in the
              Nextora prototype.
            </p>
          </div>

          <div className="nx-feature-grid">
            {features.map((feature) => {
              const Icon = feature.icon;

              return (
                <article className="nx-feature-card" key={feature.number}>
                  <div className="nx-feature-top">
                    <span className="nx-feature-number">
                      {feature.number}
                    </span>
                    <span className="nx-feature-icon">
                      <Icon size={22} />
                    </span>
                  </div>

                  <span className="nx-feature-tag">{feature.tag}</span>
                  <h3>{feature.title}</h3>
                  <p>{feature.description}</p>

                  <button
                    className="nx-feature-link"
                    onClick={
                      feature.number === "03"
                        ? openTreatment
                        : openAnalyzer
                    }
                  >
                    {feature.number === "03"
                      ? "Explore treatment support"
                      : "Explore analyzer"}
                    <ChevronRight size={17} />
                  </button>
                </article>
              );
            })}
          </div>
        </section>

        <section className="nx-workflow" id="workflow">
          <div className="nx-workflow-heading">
            <span className="nx-section-kicker">HOW IT WORKS</span>
            <h2>From image to insight.</h2>
            <p>
              A simple workflow for exploring model output while keeping
              its limitations in view.
            </p>
          </div>

          <div className="nx-steps">
            {steps.map((step, index) => (
              <article className="nx-step" key={step.number}>
                <div className="nx-step-top">
                  <span>{step.number}</span>
                  {index === 0 && <Upload size={20} />}
                  {index === 1 && <FlaskConical size={20} />}
                  {index === 2 && <Activity size={20} />}
                </div>
                <h3>{step.title}</h3>
                <p>{step.description}</p>
              </article>
            ))}
          </div>

          <button className="nx-button-primary" onClick={openAnalyzer}>
            Start exploring <ArrowRight size={18} />
          </button>
        </section>

        <section className="nx-research" id="research">
          <div className="nx-research-icon">
            <ShieldCheck size={25} />
          </div>
          <div className="nx-research-copy">
            <span className="nx-section-kicker">IMPORTANT RESEARCH NOTE</span>
            <h2>Useful exploration starts with honest limitations.</h2>
            <p>
              Nextora is a research prototype and has not been clinically
              validated. Model scores are not clinically calibrated
              probabilities. Whole-smear candidate regions are experimental
              and may not represent individual cells. Outputs cannot confirm
              or exclude malaria, measure validated parasite density, or
              replace professional laboratory assessment.
            </p>
          </div>
          <button className="nx-research-link" onClick={openTreatment}>
            Treatment-support prototype <ArrowRight size={17} />
          </button>
        </section>

        <section className="nx-final-cta">
          <div className="nx-cta-sparkle">
            <Sparkles size={23} />
          </div>
          <span className="nx-section-kicker">EXPLORE NEXTORA</span>
          <h2>
            Start with an image.
            <br />
            <span>See what the model finds.</span>
          </h2>
          <p>
            Open the workspace to explore the current research prototype.
          </p>
          <button className="nx-button-light" onClick={openAnalyzer}>
            Open the analyzer <ArrowRight size={18} />
          </button>
        </section>
      </main>

      <footer className="nx-footer">
        <a className="nx-brand nx-footer-brand" href="/">
          <span className="nx-brand-mark">
            <Microscope size={19} />
          </span>
          <span className="nx-brand-copy">
            <strong>nextora<span>.</span></strong>
            <small>AI MICROSCOPY</small>
          </span>
        </a>

        <p>Experimental research prototype. Not for clinical diagnosis.</p>

        <a href="#top" onClick={(event) => {
          event.preventDefault();
          window.scrollTo({ top: 0, behavior: "smooth" });
        }}>
          Back to top ↑
        </a>
      </footer>
    </div>
  );
}
