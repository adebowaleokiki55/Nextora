import { useEffect, useRef, useState } from "react";
import {
  Activity,
  AlertCircle,
  Camera,
  CheckCircle2,
  FileImage,
  Focus,
  Maximize2,
  Microscope,
  Minus,
  Plus,
  RotateCcw,
  ShieldCheck,
  UploadCloud,
  X,
  ChevronDown,
  ChevronUp,
} from "lucide-react";
import "./App.css";
import TreatmentSupport from "./TreatmentSupport";


const API_URL = import.meta.env.VITE_API_URL || "http://127.0.0.1:8000";
const MAX_SIZE = 10 * 1024 * 1024;
const ACCEPTED_TYPES = ["image/jpeg", "image/png", "image/webp"];
const INITIAL_CANDIDATES_VISIBLE = 5;

export default function App() {
  const inputRef = useRef(null);
  const cameraRef = useRef(null);
  const viewportRef = useRef(null);
  const dragRef = useRef(null);
  const imageRef = useRef(null);

  const [file, setFile] = useState(null);
  const [preview, setPreview] = useState("");
  const [dimensions, setDimensions] = useState(null);
  const [result, setResult] = useState(null);
  const [error, setError] = useState("");
  const [loading, setLoading] = useState(false);
  const [analysisMode, setAnalysisMode] = useState("single");

  const [zoom, setZoom] = useState(1);
  const [position, setPosition] = useState({ x: 0, y: 0 });
  const [dragging, setDragging] = useState(false);
  const [fitScale, setFitScale] = useState(1);

  // Keep the long smear candidate list collapsed by default.
  const [showAllCandidates, setShowAllCandidates] = useState(false);

  const isSmearMode = analysisMode === "smear";
  const annotatedImage = isSmearMode
    ? result?.annotated_image_base64
    : null;

  const displayedImage = annotatedImage
    ? `data:image/png;base64,${annotatedImage}`
    : preview;

  useEffect(() => {
    return () => {
      if (preview) URL.revokeObjectURL(preview);
    };
  }, [preview]);

  useEffect(() => {
    function handleKeyDown(event) {
      if (event.key === "Escape") {
        setDragging(false);
        dragRef.current = null;
      }
    }

    window.addEventListener("keydown", handleKeyDown);

    return () => {
      window.removeEventListener("keydown", handleKeyDown);
    };
  }, []);

  function chooseFile(selectedFile) {
    if (!selectedFile) return;

    setError("");
    setResult(null);
    setDimensions(null);
    setShowAllCandidates(false);

    if (!ACCEPTED_TYPES.includes(selectedFile.type)) {
      setError("Choose a JPEG, PNG, or WebP image.");
      return;
    }

    if (selectedFile.size > MAX_SIZE) {
      setError("The image must be smaller than 10 MB.");
      return;
    }

    setFile(selectedFile);
    setPreview(URL.createObjectURL(selectedFile));
    setZoom(1);
    setPosition({ x: 0, y: 0 });

    if (inputRef.current) inputRef.current.value = "";
    if (cameraRef.current) cameraRef.current.value = "";
  }

  function handleImageLoad(event) {
    const image = event.currentTarget;

    setDimensions({
      width: image.naturalWidth,
      height: image.naturalHeight,
    });

    const viewport = viewportRef.current;

    if (!viewport || !image.naturalWidth || !image.naturalHeight) {
      return;
    }

    const scale = Math.min(
      (viewport.clientWidth - 40) / image.naturalWidth,
      (viewport.clientHeight - 40) / image.naturalHeight,
      1
    );

    setFitScale(scale || 1);
    setZoom(1);
    setPosition({ x: 0, y: 0 });
  }

  function resetView() {
    setZoom(1);
    setPosition({ x: 0, y: 0 });
  }

  function changeZoom(amount) {
    setZoom((current) => Math.max(1, Math.min(5, current + amount)));
  }

  function handleWheel(event) {
    if (!displayedImage) return;

    event.preventDefault();
    changeZoom(event.deltaY < 0 ? 0.2 : -0.2);
  }

  function handlePointerDown(event) {
    if (!displayedImage || zoom <= 1) return;

    event.currentTarget.setPointerCapture(event.pointerId);

    dragRef.current = {
      x: event.clientX,
      y: event.clientY,
      startX: position.x,
      startY: position.y,
    };

    setDragging(true);
  }

  function handlePointerMove(event) {
    if (!dragRef.current) return;

    const drag = dragRef.current;

    setPosition({
      x: drag.startX + event.clientX - drag.x,
      y: drag.startY + event.clientY - drag.y,
    });
  }

  function handlePointerUp() {
    dragRef.current = null;
    setDragging(false);
  }

  function selectMode(mode) {
    if (loading) return;

    setAnalysisMode(mode);
    setResult(null);
    setError("");
    setShowAllCandidates(false);
    resetView();
  }

  function clearCurrent() {
    setFile(null);
    setPreview("");
    setDimensions(null);
    setResult(null);
    setError("");
    setZoom(1);
    setPosition({ x: 0, y: 0 });
    setDragging(false);
    setShowAllCandidates(false);
    dragRef.current = null;

    if (inputRef.current) inputRef.current.value = "";
    if (cameraRef.current) cameraRef.current.value = "";
  }

  async function analyzeImage() {
    if (!file || loading) return;

    setLoading(true);
    setError("");
    setResult(null);
    setShowAllCandidates(false);

    const formData = new FormData();
    formData.append("file", file);

    const endpoint = isSmearMode
      ? "/api/analyze-smear"
      : "/api/predict";

    try {
      const response = await fetch(`${API_URL}${endpoint}`, {
        method: "POST",
        body: formData,
      });

      const data = await response.json();

      if (!response.ok) {
        throw new Error(data.detail || "Analysis request failed.");
      }

      if (isSmearMode && !data.annotated_image_base64) {
        throw new Error(
          "The server returned no annotated image. Check the backend response."
        );
      }

      setResult(data);
    } catch (err) {
      setError(
        err.message === "Failed to fetch"
          ? "Cannot connect to Nextora. Check your internet connection and backend URL."
          : err.message || "Something went wrong. Please try again."
      );
    } finally {
      setLoading(false);
    }
  }

  const isParasitized = result?.predicted_class === 0;

  const parasiteScore = Number(
    result?.probabilities?.parasitized ?? 0
  );

  const uninfectedScore = Number(
    result?.probabilities?.uninfected ?? 0
  );

  const smearCells = Array.isArray(result?.cells) ? result.cells : [];

  const visibleSmearCells = showAllCandidates
    ? smearCells
    : smearCells.slice(0, INITIAL_CANDIDATES_VISIBLE);

  return (
    <div className="app-shell explorer-shell">
      <header className="topbar">
        <a className="brand" href="/" aria-label="Nextora home">
          <span className="brand-icon">
            <Microscope size={23} />
          </span>
          <span>
            <strong>Nextora</strong>
            <small>AI MICROSCOPY PLATFORM</small>
          </span>
        </a>

        <div className="topbar-right">
          <span className="workspace-label">
            <Activity size={15} /> Cell Explorer
          </span>

          <span className="status-pill">
            <span className="status-dot" />
            Research prototype
          </span>
        </div>
      </header>

      <main className="explorer-main">
        <section className="explorer-intro">
          <div>
            <span className="eyebrow">
              <Activity size={15} /> COMPUTER VISION / MICROSCOPY
            </span>

            <h1>
              Cell <span>Explorer.</span>
            </h1>

            <p>
              Inspect microscopy images, explore model outputs, and examine
              candidate regions across blood-cell imagery in one workspace.
            </p>
          </div>

          <div className="model-chip">
            <span className="model-chip-dot" />
            <span>
              <strong>PlasmoSENet</strong>
              <small>
                {isSmearMode
                  ? "Experimental smear analysis"
                  : "Binary classification model"}
              </small>
            </span>
          </div>
        </section>

        <div className="explorer-grid">
          <section className="viewer-panel">
            <div className="viewer-heading">
              <div>
                <span className="step-label">IMAGE WORKSPACE</span>
                <h2>Microscopy viewer</h2>
              </div>

              {preview && (
                <button
                  className="viewer-icon-button"
                  onClick={clearCurrent}
                  disabled={loading}
                  title="Remove image"
                  aria-label="Remove image"
                >
                  <X size={17} />
                </button>
              )}
            </div>

            <div
              className={`microscope-viewport ${
                displayedImage ? "has-image" : ""
              } ${dragging ? "is-dragging" : ""}`}
              ref={viewportRef}
              onWheel={handleWheel}
              onPointerDown={handlePointerDown}
              onPointerMove={handlePointerMove}
              onPointerUp={handlePointerUp}
              onPointerCancel={handlePointerUp}
            >
              {displayedImage ? (
                <img
                  ref={imageRef}
                  key={
                    displayedImage.startsWith("data:image/png;base64,")
                      ? "annotated"
                      : preview
                  }
                  className="microscopy-image"
                  src={displayedImage}
                  alt={
                    annotatedImage
                      ? "Microscopy image annotated with experimental candidate predictions"
                      : "Selected microscopy sample"
                  }
                  draggable="false"
                  onLoad={handleImageLoad}
                  style={{
                    transform: `translate(${position.x}px, ${position.y}px) scale(${zoom})`,
                    cursor:
                      zoom > 1
                        ? dragging
                          ? "grabbing"
                          : "grab"
                        : "default",
                  }}
                />
              ) : (
                <div className="viewer-empty">
                  <div className="viewer-empty-icon">
                    <Microscope size={32} />
                  </div>

                  <h3>Your workspace is ready</h3>

                  <p>
                    Load a microscopy image to begin visual inspection.
                  </p>

                  <button
                    className="browse-button"
                    onClick={() => inputRef.current?.click()}
                  >
                    <UploadCloud size={16} /> Choose image
                  </button>

                  <button
                    className="viewer-camera-link"
                    onClick={() => cameraRef.current?.click()}
                  >
                    <Camera size={16} /> Take a photo
                  </button>
                </div>
              )}

              {displayedImage && (
                <div className="viewer-overlay-top">
                  <span className="image-status">
                    <span />
                    {annotatedImage ? "ANNOTATED RESULTS" : "IMAGE LOADED"}
                  </span>

                  <span className="zoom-label">
                    {Math.round(zoom * 100)}%
                  </span>
                </div>
              )}

              {displayedImage && (
                <div
                  className="viewer-controls"
                  onPointerDown={(event) => event.stopPropagation()}
                >
                  <button
                    onClick={() => changeZoom(-0.25)}
                    disabled={zoom <= 1}
                    aria-label="Zoom out"
                    title="Zoom out"
                  >
                    <Minus size={17} />
                  </button>

                  <span>{Math.round(zoom * 100)}%</span>

                  <button
                    onClick={() => changeZoom(0.25)}
                    disabled={zoom >= 5}
                    aria-label="Zoom in"
                    title="Zoom in"
                  >
                    <Plus size={17} />
                  </button>

                  <span className="control-divider" />

                  <button
                    onClick={resetView}
                    aria-label="Reset view"
                    title="Reset view"
                  >
                    <RotateCcw size={16} />
                  </button>

                  <button
                    onClick={() => {
                      setZoom(1);
                      setPosition({ x: 0, y: 0 });
                    }}
                    aria-label="Fit image"
                    title="Fit image"
                  >
                    <Focus size={16} />
                  </button>
                </div>
              )}
            </div>

            {preview ? (
              <>
                <div className="image-metadata">
                  <div className="metadata-icon">
                    <FileImage size={18} />
                  </div>

                  <div className="metadata-main">
                    <strong title={file?.name}>
                      {file?.name || "Captured image"}
                    </strong>

                    <span>
                      {dimensions
                        ? `${dimensions.width} × ${dimensions.height} pixels`
                        : "Reading image dimensions…"}
                      {file
                        ? ` · ${(file.size / 1024).toFixed(1)} KB`
                        : ""}
                    </span>
                  </div>

                  <button
                    className="text-button"
                    onClick={() => inputRef.current?.click()}
                    disabled={loading}
                  >
                    Change image
                  </button>
                </div>

                <div className="viewer-tip">
                  <Maximize2 size={14} />
                  <span>
                    Scroll to zoom. Drag the image to inspect details when
                    zoomed in.
                  </span>
                </div>
              </>
            ) : (
              <div className="upload-alternatives">
                <button
                  className="browse-button"
                  onClick={() => inputRef.current?.click()}
                >
                  <FileImage size={16} /> Browse files
                </button>

                <button
                  className="camera-outline-button"
                  onClick={() => cameraRef.current?.click()}
                >
                  <Camera size={16} /> Capture image
                </button>
              </div>
            )}

            <input
              ref={inputRef}
              type="file"
              accept="image/jpeg,image/png,image/webp"
              hidden
              onChange={(event) => chooseFile(event.target.files?.[0])}
            />

            <input
              ref={cameraRef}
              type="file"
              accept="image/jpeg,image/png,image/webp"
              capture="environment"
              hidden
              onChange={(event) => chooseFile(event.target.files?.[0])}
            />

            {error && (
              <div className="error-box" role="alert">
                <AlertCircle size={18} />
                <span>{error}</span>
              </div>
            )}

            <div className="analysis-mode">
              <span className="step-label">ANALYSIS MODE</span>

              <div className="mode-options">
                <button
                  type="button"
                  className={`mode-option ${
                    analysisMode === "single" ? "active" : ""
                  }`}
                  onClick={() => selectMode("single")}
                  disabled={loading}
                >
                  Single Cell
                </button>

                <button
                  type="button"
                  className={`mode-option ${
                    analysisMode === "smear" ? "active" : ""
                  }`}
                  onClick={() => selectMode("smear")}
                  disabled={loading}
                >
                  Whole Smear
                </button>
              </div>

              <p>
                {isSmearMode
                  ? "Explore candidate regions and their model classifications across a microscopy image."
                  : "Classify an isolated cell image using PlasmoSENet."}
              </p>
            </div>

            <button
              className="primary-button"
              disabled={!file || loading}
              onClick={analyzeImage}
            >
              {loading ? (
                <>
                  <span className="spinner" />
                  Analyzing sample…
                </>
              ) : (
                <>
                  <Microscope size={18} />
                  {isSmearMode
                    ? "Analyze whole smear"
                    : "Analyze cell image"}
                </>
              )}
            </button>

            <p className="privacy-note">
              <ShieldCheck size={15} />
              Images are sent to your configured backend for processing.
            </p>
          </section>

          <aside className="analysis-sidebar">
            <section className="analysis-card">
              <div className="sidebar-heading">
                <div>
                  <span className="step-label">MODEL OUTPUT</span>
                  <h2>Analysis</h2>
                </div>

                <Activity size={20} className="muted-icon" />
              </div>

              {!result && !loading && (
                <div className="analysis-empty">
                  <div className="analysis-empty-icon">
                    <Activity size={24} />
                  </div>

                  <strong>Awaiting sample</strong>

                  <p>
                    Load an image and run analysis to see the model output
                    here.
                  </p>
                </div>
              )}

              {loading && (
                <div className="analysis-empty">
                  <span className="large-spinner" />
                  <strong>
                    {isSmearMode
                      ? "Analyzing microscopy field"
                      : "Processing cell image"}
                  </strong>

                  <p>
                    {isSmearMode
                      ? "Finding candidate regions and classifying their image crops."
                      : "PlasmoSENet is running inference."}
                  </p>
                </div>
              )}

              {result && analysisMode === "single" && (
                <div className="result-content" aria-live="polite">
                  <div
                    className={`prediction-card ${
                      isParasitized ? "infected" : "uninfected"
                    }`}
                  >
                    <span className="prediction-symbol">
                      {isParasitized ? (
                        <AlertCircle size={23} />
                      ) : (
                        <CheckCircle2 size={23} />
                      )}
                    </span>

                    <div>
                      <span className="result-overline">PREDICTED CLASS</span>
                      <h3>{result.prediction}</h3>
                      <p>Classification of the supplied cell image</p>
                    </div>
                  </div>

                  <div className="score-heading">
                    <h3>Class score breakdown</h3>
                    <span>Softmax output</span>
                  </div>

                  <div className="score-row">
                    <div className="score-label">
                      <span>Parasitized</span>
                      <strong>{parasiteScore.toFixed(2)}%</strong>
                    </div>

                    <div className="score-track">
                      <div
                        className="score-fill parasite-fill"
                        style={{
                          width: `${Math.max(
                            0,
                            Math.min(100, parasiteScore)
                          )}%`,
                        }}
                      />
                    </div>
                  </div>

                  <div className="score-row">
                    <div className="score-label">
                      <span>Uninfected</span>
                      <strong>{uninfectedScore.toFixed(2)}%</strong>
                    </div>

                    <div className="score-track">
                      <div
                        className="score-fill healthy-fill"
                        style={{
                          width: `${Math.max(
                            0,
                            Math.min(100, uninfectedScore)
                          )}%`,
                        }}
                      />
                    </div>
                  </div>

                  <div className="confidence-box">
                    <span>Top-class model score</span>
                    <strong>
                      {Number(result.confidence).toFixed(2)}%
                    </strong>
                  </div>

                  <div className="disclaimer">
                    <AlertCircle size={17} />
                    <p>
                      Research prototype only. Scores are not clinically
                      calibrated probabilities and cannot diagnose or rule
                      out malaria. Seek qualified laboratory assessment.
                    </p>
                  </div>

                  <button
                    type="button"
                    className="secondary-button"
                    onClick={clearCurrent}
                  >
                    <RotateCcw size={16} /> Analyze another image
                  </button>
                </div>
              )}

              {result && analysisMode === "smear" && (
                <div className="result-content smear-results" aria-live="polite">
                  <div className="smear-success-heading">
                    <CheckCircle2 size={20} />

                    <div>
                      <strong>Smear analysis complete</strong>
                      <p>Experimental candidate-region analysis</p>
                    </div>
                  </div>

                  <div className="smear-summary-grid">
                    <div className="smear-stat">
                      <span>Candidate regions</span>
                      <strong>{result.candidate_count}</strong>
                    </div>

                    <div className="smear-stat infected-stat">
                      <span>Classified infected</span>
                      <strong>{result.candidate_infected_count}</strong>
                    </div>

                    <div className="smear-stat healthy-stat">
                      <span>Classified uninfected</span>
                      <strong>{result.candidate_uninfected_count}</strong>
                    </div>
                  </div>

                  <div className="score-heading candidate-section-heading">
                    <h3>Candidate predictions</h3>
                    <span>{smearCells.length} regions</span>
                  </div>

                  {smearCells.length === 0 ? (
                    <div className="no-candidates">
                      No candidate regions were detected. This does not
                      establish that the image is free of infected cells.
                    </div>
                  ) : (
                    <>
                      <div className="candidate-list">
                        {visibleSmearCells.map((cell) => (
                          <div className="candidate-item" key={cell.cell_id}>
                            <div className="candidate-topline">
                              <strong>Candidate #{cell.cell_id}</strong>

                              <span
                                className={`candidate-badge ${
                                  cell.predicted_class === 0
                                    ? "infected-badge"
                                    : "healthy-badge"
                                }`}
                              >
                                {cell.predicted_class === 0
                                  ? "Infected class"
                                  : "Uninfected class"}
                              </span>
                            </div>

                            <div className="candidate-details">
                              <span>Model score</span>
                              <strong>
                                {Number(cell.confidence).toFixed(2)}%
                              </strong>
                            </div>

                            <div className="candidate-details">
                              <span>Region size</span>
                              <span>
                                {cell.box.width} × {cell.box.height} px
                              </span>
                            </div>

                            <div className="candidate-details">
                              <span>Position</span>
                              <span>
                                x: {cell.box.x}, y: {cell.box.y}
                              </span>
                            </div>
                          </div>
                        ))}
                      </div>

                      {smearCells.length > INITIAL_CANDIDATES_VISIBLE && (
                        <button
                          type="button"
                          className="candidate-toggle-button"
                          onClick={() =>
                            setShowAllCandidates((current) => !current)
                          }
                          aria-expanded={showAllCandidates}
                        >
                          {showAllCandidates ? (
                            <>
                              <ChevronUp size={17} />
                              Show fewer candidates
                            </>
                          ) : (
                            <>
                              <ChevronDown size={17} />
                              View all {smearCells.length} candidate regions
                            </>
                          )}
                        </button>
                      )}
                    </>
                  )}

                  <div className="disclaimer smear-disclaimer">
                    <AlertCircle size={17} />

                    <p>
                      Experimental only. Candidate regions may not correspond
                      to individual cells. Model scores are not clinically
                      calibrated, and these counts are not a validated
                      parasitemia measurement or a malaria diagnosis.
                    </p>
                  </div>

                  <button
                    type="button"
                    className="secondary-button"
                    onClick={clearCurrent}
                  >
                    <RotateCcw size={16} /> Analyze another image
                  </button>
                </div>
              )}
            </section>

            <section className="sample-info-card">
              <span className="step-label">ABOUT THIS WORKSPACE</span>

              <div className="info-line">
                <span>Model</span>
                <strong>PlasmoSENet</strong>
              </div>

              <div className="info-line">
                <span>Task</span>
                <strong>
                  {isSmearMode
                    ? "Experimental candidate analysis"
                    : "Binary cell classification"}
                </strong>
              </div>

              <div className="info-line">
                <span>Input format</span>
                <strong>JPEG · PNG · WebP</strong>
              </div>

              <div className="info-line">
                <span>Maximum file size</span>
                <strong>10 MB</strong>
              </div>

              <p>
                {isSmearMode
                  ? "Candidate regions are identified using image-processing heuristics. Detection and classification have not been validated for clinical use."
                  : "This mode classifies an isolated cell image. It does not automatically identify or outline parasites."}
              </p>
            </section>
          </aside>
        </div>

        {/* Treatment support now appears before the page footer. */}
        <section className="treatment-support-section" id="treatment-support">
          <TreatmentSupport />
        </section>

        <footer className="page-footer">
          <span>Nextora · Powered by PlasmoSENet</span>
          <span>Research prototype · Not for clinical diagnosis</span>
        </footer>
      </main>
    </div>
  );
}