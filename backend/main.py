import os

# Set these BEFORE importing PyTorch, OpenCV, or numerical libraries.
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"
os.environ["OPENBLAS_NUM_THREADS"] = "1"
os.environ["NUMEXPR_NUM_THREADS"] = "1"

import base64
import json
import threading
from io import BytesIO
from pathlib import Path
from typing import Any, Literal

import cv2
import requests
import torch
from PIL import Image, UnidentifiedImageError
from torchvision import transforms
from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

from plasmosenet import PlasmoSENet
from backend.smear_analysis import analyze_smear
from backend.treatment_engine import (
    TreatmentRequest,
    assess_treatment,
)


# ==================================================
# RESOURCE LIMITS AND CONFIGURATION
# ==================================================

PROJECT_ROOT = Path(__file__).resolve().parent.parent
MODEL_PATH = PROJECT_ROOT / "model.pth"

MAX_UPLOAD_BYTES = 10 * 1024 * 1024

# Reject images with excessive dimensions before expensive processing.
MAX_IMAGE_PIXELS = 12_000_000

# Only one request at a time may run local model inference.
# This helps prevent multiple image requests from increasing peak RAM.
INFERENCE_LOCK = threading.BoundedSemaphore(value=1)

# Keep CPU thread usage low on the Render free instance.
torch.set_num_threads(1)

try:
    torch.set_num_interop_threads(1)
except RuntimeError:
    # PyTorch may already have initialized its inter-op thread pool.
    pass

cv2.setNumThreads(0)

OLLAMA_URL = os.getenv(
    "OLLAMA_URL",
    "https://ollama.com/api/chat",
).rstrip("/")

OLLAMA_MODEL = os.getenv(
    "OLLAMA_MODEL",
    "gpt-oss:20b",
)

OLLAMA_API_KEY = os.getenv("OLLAMA_API_KEY")

CLASS_NAMES = {
    0: "Parasitized (Infected)",
    1: "Uninfected",
}

DEVICE = torch.device(
    "cuda" if torch.cuda.is_available() else "cpu"
)


# ==================================================
# FASTAPI APPLICATION
# ==================================================

app = FastAPI(
    title="Nextora AI API",
    version="1.6.0",
    description=(
        "Malaria microscopy research prototype with image "
        "classification, a tool-using research assistant, and "
        "preliminary treatment decision support."
    ),
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "http://localhost:5173",
        "http://127.0.0.1:5173",
        "https://nextora-a5yj6qcjy-adebowale-okikis-projects.vercel.app",
        "https://nextora-zeta.vercel.app",
    ],
    allow_credentials=False,
    allow_methods=["GET", "POST"],
    allow_headers=["*"],
)


# ==================================================
# MEMORY DIAGNOSTICS
# ==================================================

def read_process_memory() -> dict:
    """
    Read Linux process memory information without adding psutil.

    Render runs Linux. If /proc is unavailable, return a diagnostic
    error instead of causing the health endpoint to fail.
    """
    result = {
        "rss_mb": None,
        "peak_rss_mb": None,
        "threads": None,
        "cgroup_memory_current_mb": None,
        "cgroup_memory_limit_mb": None,
    }

    try:
        with open("/proc/self/status", "r", encoding="utf-8") as file:
            for line in file:
                if line.startswith("VmRSS:"):
                    result["rss_mb"] = round(
                        int(line.split()[1]) / 1024,
                        2,
                    )
                elif line.startswith("VmHWM:"):
                    result["peak_rss_mb"] = round(
                        int(line.split()[1]) / 1024,
                        2,
                    )
                elif line.startswith("Threads:"):
                    result["threads"] = int(line.split()[1])
    except (OSError, ValueError, IndexError):
        pass

    # cgroup v2 memory counters, commonly available on Linux hosts.
    try:
        with open(
            "/sys/fs/cgroup/memory.current",
            "r",
            encoding="utf-8",
        ) as file:
            result["cgroup_memory_current_mb"] = round(
                int(file.read().strip()) / (1024 * 1024),
                2,
            )
    except (OSError, ValueError):
        pass

    try:
        with open(
            "/sys/fs/cgroup/memory.max",
            "r",
            encoding="utf-8",
        ) as file:
            value = file.read().strip()

            if value != "max":
                result["cgroup_memory_limit_mb"] = round(
                    int(value) / (1024 * 1024),
                    2,
                )
    except (OSError, ValueError):
        pass

    return result


@app.get("/api/diagnostics/memory")
def memory_diagnostics():
    """
    Report process RAM usage and current inference-lock status.
    This endpoint does not run model inference.
    """
    return {
        "status": "ok",
        "memory": read_process_memory(),
        "inference_busy": INFERENCE_LOCK.locked(),
        "device": str(DEVICE),
    }


# ==================================================
# IMAGE PREPROCESSING
# ==================================================

transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225],
    ),
])


def validate_image_dimensions(image_bytes: bytes) -> None:
    """Reject invalid or excessively large images before inference."""
    try:
        with Image.open(BytesIO(image_bytes)) as image:
            width, height = image.size

            if width <= 0 or height <= 0:
                raise HTTPException(
                    status_code=400,
                    detail="The image dimensions are invalid.",
                )

            if width * height > MAX_IMAGE_PIXELS:
                raise HTTPException(
                    status_code=413,
                    detail=(
                        "The image dimensions are too large. "
                        "Please upload an image containing no more "
                        "than 12 million pixels."
                    ),
                )

            image.verify()

    except HTTPException:
        raise

    except (
        UnidentifiedImageError,
        OSError,
        ValueError,
    ) as exc:
        raise HTTPException(
            status_code=400,
            detail="The uploaded file could not be processed as an image.",
        ) from exc


def acquire_inference_slot() -> None:
    """
    Avoid queuing multiple image requests in memory.

    Return a retryable response if another request is already using
    the model rather than letting concurrent requests pile up.
    """
    if not INFERENCE_LOCK.acquire(blocking=False):
        raise HTTPException(
            status_code=503,
            detail=(
                "The image-analysis service is busy. "
                "Please wait a few seconds and try again."
            ),
            headers={"Retry-After": "5"},
        )


# ==================================================
# LOAD MODEL
# ==================================================

def load_model():
    if not MODEL_PATH.exists():
        raise RuntimeError(
            f"Model weights not found: {MODEL_PATH}"
        )

    loaded_model = PlasmoSENet(num_classes=2)

    state_dict = torch.load(
        MODEL_PATH,
        map_location=DEVICE,
        weights_only=True,
    )

    loaded_model.load_state_dict(state_dict)
    loaded_model.to(DEVICE)
    loaded_model.eval()

    return loaded_model


model = load_model()


# ==================================================
# HEALTH CHECK
# ==================================================

@app.get("/api/health")
def health():
    return {
        "status": "ok",
        "model_loaded": model is not None,
        "device": str(DEVICE),
        "agent_model": OLLAMA_MODEL,
        "agent_runtime": "Ollama Cloud",
        "agent_configured": bool(
            OLLAMA_API_KEY and OLLAMA_MODEL
        ),
        "clinical_validation": False,
    }


# ==================================================
# SINGLE-CELL PREDICTION
# ==================================================

@app.post("/api/predict")
def predict(file: UploadFile = File(...)):
    if file.content_type not in {
        "image/jpeg",
        "image/png",
        "image/webp",
    }:
        raise HTTPException(
            status_code=415,
            detail="Upload a JPEG, PNG, or WebP image.",
        )

    image_bytes = file.file.read(MAX_UPLOAD_BYTES + 1)

    if not image_bytes:
        raise HTTPException(
            status_code=400,
            detail="The file is empty.",
        )

    if len(image_bytes) > MAX_UPLOAD_BYTES:
        raise HTTPException(
            status_code=413,
            detail="Image exceeds the 10 MB upload limit.",
        )

    validate_image_dimensions(image_bytes)

    acquire_inference_slot()

    try:
        with Image.open(BytesIO(image_bytes)) as source:
            image = source.convert("RGB")

        input_tensor = (
            transform(image)
            .unsqueeze(0)
            .to(DEVICE)
        )

        with torch.inference_mode():
            logits = model(input_tensor)
            probabilities = torch.softmax(logits, dim=1)[0]

        if probabilities.numel() != 2:
            raise RuntimeError(
                "Unexpected model output shape."
            )

        predicted_class = int(
            torch.argmax(probabilities).item()
        )

        confidence = float(
            probabilities[predicted_class].item()
        )

        return {
            "predicted_class": predicted_class,
            "prediction": CLASS_NAMES[predicted_class],
            "confidence": round(confidence * 100, 2),
            "probabilities": {
                "parasitized": round(
                    float(probabilities[0]) * 100, 2
                ),
                "uninfected": round(
                    float(probabilities[1]) * 100, 2
                ),
            },
            "model": "PlasmoSENet",
            "disclaimer": (
                "Research screening result for an isolated cell image; "
                "not a clinical diagnosis."
            ),
        }

    except HTTPException:
        raise

    except (
        UnidentifiedImageError,
        OSError,
        ValueError,
    ) as exc:
        raise HTTPException(
            status_code=400,
            detail=(
                "The uploaded file could not be processed as an image."
            ),
        ) from exc

    except Exception as exc:
        print(
            f"Prediction error: {type(exc).__name__}: {exc}"
        )

        raise HTTPException(
            status_code=500,
            detail="Prediction failed. Check the backend logs.",
        ) from exc

    finally:
        INFERENCE_LOCK.release()


# ==================================================
# WHOLE-SMEAR ANALYSIS
# ==================================================

@app.post("/api/analyze-smear")
async def analyze_smear_endpoint(
    file: UploadFile = File(...),
):
    allowed_types = {
        "image/jpeg",
        "image/png",
        "image/webp",
    }

    if file.content_type not in allowed_types:
        raise HTTPException(
            status_code=400,
            detail=(
                "Upload a JPEG, PNG, or WebP microscopy image."
            ),
        )

    image_bytes = await file.read(MAX_UPLOAD_BYTES + 1)

    if not image_bytes:
        raise HTTPException(
            status_code=400,
            detail="The uploaded image is empty.",
        )

    if len(image_bytes) > MAX_UPLOAD_BYTES:
        raise HTTPException(
            status_code=413,
            detail="Image must be 10 MB or smaller.",
        )

    validate_image_dimensions(image_bytes)
    acquire_inference_slot()

    try:
        result = analyze_smear(
            image_bytes=image_bytes,
            model=model,
            device=DEVICE,
        )

        annotated_image = result.pop("annotated_image")

        result["annotated_image_base64"] = (
            base64.b64encode(annotated_image)
            .decode("ascii")
        )

        result["annotated_image_mime_type"] = "image/png"

        return result

    except HTTPException:
        raise

    except (ValueError, cv2.error) as exc:
        raise HTTPException(
            status_code=400,
            detail=f"Could not analyze this image: {exc}",
        ) from exc

    except Exception as exc:
        print(
            f"Smear analysis error: {type(exc).__name__}: {exc}"
        )

        raise HTTPException(
            status_code=500,
            detail="Smear analysis failed. Check the backend logs.",
        ) from exc

    finally:
        INFERENCE_LOCK.release()


# ==================================================
# ANALYSIS AGENT REQUEST SCHEMA
# ==================================================

class AgentExplainRequest(BaseModel):
    mode: Literal["single_cell", "whole_smear"]
    analysis: dict[str, Any]


# ==================================================
# TOOL 1: INSPECT EVIDENCE
# ==================================================

def inspect_analysis_evidence(
    mode: str,
    analysis: dict,
) -> dict:
    """Summarize actual outputs supplied by Nextora."""

    if mode == "single_cell":
        required_fields = [
            "predicted_class",
            "prediction",
            "confidence",
            "probabilities",
            "model",
        ]

        missing_fields = [
            field
            for field in required_fields
            if field not in analysis
        ]

        return {
            "mode": mode,
            "prediction": analysis.get("prediction"),
            "predicted_class": analysis.get("predicted_class"),
            "confidence": analysis.get("confidence"),
            "probabilities": analysis.get("probabilities"),
            "model": analysis.get("model"),
            "missing_fields": missing_fields,
            "disclaimer": analysis.get("disclaimer"),
        }

    cells = analysis.get("cells", [])

    if not isinstance(cells, list):
        cells = []

    infected = sum(
        1
        for cell in cells
        if isinstance(cell, dict)
        and cell.get("predicted_class") == 0
    )

    uninfected = sum(
        1
        for cell in cells
        if isinstance(cell, dict)
        and cell.get("predicted_class") == 1
    )

    return {
        "mode": mode,
        "image_width": analysis.get("image_width"),
        "image_height": analysis.get("image_height"),
        "reported_candidate_count": analysis.get(
            "candidate_count"
        ),
        "reported_infected_count": analysis.get(
            "candidate_infected_count"
        ),
        "reported_uninfected_count": analysis.get(
            "candidate_uninfected_count"
        ),
        "cells_returned": len(cells),
        "counts_from_returned_cell_records": {
            "infected_class": infected,
            "uninfected_class": uninfected,
        },
        "warning": (
            "Candidate regions are heuristic detections and are not "
            "necessarily confirmed individual cells."
        ),
    }


# ==================================================
# TOOL 2: CHECK UNCERTAINTY AND CONSISTENCY
# ==================================================

def check_analysis_uncertainty(
    mode: str,
    analysis: dict,
) -> dict:
    """Run deterministic checks, not clinical validation."""

    issues = []
    checks = {}

    if mode == "single_cell":
        prediction = analysis.get("prediction")
        confidence = analysis.get("confidence")
        probabilities = analysis.get("probabilities")

        if (
            not isinstance(confidence, (int, float))
            or isinstance(confidence, bool)
        ):
            issues.append(
                "Confidence is missing or not numeric."
            )
        elif not 0 <= confidence <= 100:
            issues.append(
                "Confidence is outside the 0–100 range."
            )
        else:
            checks["confidence_in_valid_range"] = True

        if not isinstance(probabilities, dict):
            issues.append("Class probabilities are missing.")
        else:
            p_infected = probabilities.get("parasitized")
            p_uninfected = probabilities.get("uninfected")

            valid_probabilities = all(
                isinstance(value, (int, float))
                and not isinstance(value, bool)
                and 0 <= value <= 100
                for value in [p_infected, p_uninfected]
            )

            if not valid_probabilities:
                issues.append(
                    "Class probabilities are missing or invalid."
                )
            else:
                total = p_infected + p_uninfected

                checks["probability_total_percent"] = round(
                    total, 2
                )

                checks["probabilities_sum_to_100"] = (
                    abs(total - 100) <= 1.0
                )

                if abs(total - 100) > 1.0:
                    issues.append(
                        "Class probabilities do not sum to approximately 100%."
                    )

                predicted_class = analysis.get("predicted_class")

                expected_confidence = (
                    p_infected
                    if predicted_class == 0
                    else p_uninfected
                    if predicted_class == 1
                    else None
                )

                if (
                    isinstance(confidence, (int, float))
                    and not isinstance(confidence, bool)
                    and expected_confidence is not None
                ):
                    matches = (
                        abs(confidence - expected_confidence) <= 1.0
                    )

                    checks[
                        "confidence_matches_predicted_class"
                    ] = matches

                    if not matches:
                        issues.append(
                            "Confidence does not match the probability "
                            "for the predicted class."
                        )

        if prediction not in {
            "Parasitized (Infected)",
            "Uninfected",
        }:
            issues.append(
                "Prediction label is missing or unrecognized."
            )

        checks["clinical_calibration_established"] = False

        issues.append(
            "The model's confidence has not been established as a "
            "calibrated clinical probability."
        )

    else:
        candidate_count = analysis.get("candidate_count")
        cells = analysis.get("cells", [])

        if (
            not isinstance(candidate_count, int)
            or isinstance(candidate_count, bool)
            or candidate_count < 0
        ):
            issues.append(
                "Candidate-region count is missing or invalid."
            )

        if not isinstance(cells, list):
            issues.append(
                "Cell result records are missing or invalid."
            )
        else:
            checks["returned_cell_records"] = len(cells)

            if (
                isinstance(candidate_count, int)
                and not isinstance(candidate_count, bool)
                and candidate_count != len(cells)
            ):
                issues.append(
                    "Reported candidate count differs from the number "
                    "of returned cell records. Results may be filtered "
                    "or truncated."
                )

        checks["candidate_regions_confirmed_as_cells"] = False
        checks["parasitemia_validated"] = False

        issues.extend([
            (
                "Whole-smear candidate regions are heuristic and may "
                "not represent confirmed individual cells."
            ),
            (
                "Candidate counts are not a validated parasitemia "
                "measurement."
            ),
        ])

    checks["clinical_validation"] = False

    return {
        "mode": mode,
        "checks": checks,
        "issues": issues,
        "interpretation": (
            "These are software consistency and evidence-completeness "
            "checks. They do not validate the model clinically or "
            "establish whether a patient has malaria."
        ),
    }


# ==================================================
# APPROVED ANALYSIS AGENT TOOLS
# ==================================================

AGENT_TOOLS = [
    {
        "type": "function",
        "function": {
            "name": "inspect_analysis_evidence",
            "description": (
                "Inspect the supplied Nextora analysis and summarize "
                "the actual prediction evidence. Call before reporting."
            ),
            "parameters": {
                "type": "object",
                "properties": {},
                "required": [],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "check_analysis_uncertainty",
            "description": (
                "Check missing fields, confidence validity, probability "
                "consistency, and known limitations. Call before reporting."
            ),
            "parameters": {
                "type": "object",
                "properties": {},
                "required": [],
            },
        },
    },
]


# ==================================================
# OLLAMA CLOUD CLIENT
# ==================================================

def ask_ollama(
    messages: list[dict],
    tools: list[dict] | None = None,
) -> dict:
    """Send a chat request to Ollama Cloud."""

    if not OLLAMA_API_KEY:
        raise HTTPException(
            status_code=503,
            detail=(
                "Ollama Cloud API key is not configured. "
                "Set OLLAMA_API_KEY in the backend environment."
            ),
        )

    if not OLLAMA_MODEL:
        raise HTTPException(
            status_code=503,
            detail="OLLAMA_MODEL is not configured.",
        )

    payload = {
        "model": OLLAMA_MODEL,
        "messages": messages,
        "tools": AGENT_TOOLS if tools is None else tools,
        "stream": False,
        "options": {
            "temperature": 0,
            "num_predict": 500,
        },
    }

    try:
        response = requests.post(
            OLLAMA_URL,
            headers={
                "Authorization": f"Bearer {OLLAMA_API_KEY}",
                "Content-Type": "application/json",
            },
            json=payload,
            timeout=180,
        )

        response.raise_for_status()
        data = response.json()

    except requests.HTTPError as exc:
        status_code = (
            exc.response.status_code
            if exc.response is not None
            else None
        )

        print(
            "Ollama Cloud HTTP error:",
            status_code,
            exc.response.text[:1000]
            if exc.response is not None
            else str(exc),
        )

        if status_code in (401, 403):
            raise HTTPException(
                status_code=502,
                detail=(
                    "Ollama Cloud authentication failed. Check the "
                    "API key and model access in Render."
                ),
            ) from exc

        if status_code == 429:
            raise HTTPException(
                status_code=503,
                detail=(
                    "Ollama Cloud rate limit or usage limit reached. "
                    "Check your Ollama account."
                ),
            ) from exc

        if status_code == 404:
            raise HTTPException(
                status_code=502,
                detail=(
                    "Ollama Cloud could not find the requested model "
                    "or API endpoint. Verify OLLAMA_URL and OLLAMA_MODEL."
                ),
            ) from exc

        raise HTTPException(
            status_code=502,
            detail=(
                "Ollama Cloud returned an HTTP error. "
                "Check the backend logs."
            ),
        ) from exc

    except requests.Timeout as exc:
        raise HTTPException(
            status_code=504,
            detail="Ollama Cloud timed out.",
        ) from exc

    except requests.ConnectionError as exc:
        raise HTTPException(
            status_code=503,
            detail=(
                "Cannot connect to Ollama Cloud. Check the API URL "
                "and the backend network connection."
            ),
        ) from exc

    except requests.RequestException as exc:
        print(
            f"Ollama Cloud request error: "
            f"{type(exc).__name__}: {exc}"
        )

        raise HTTPException(
            status_code=502,
            detail="Ollama Cloud request failed.",
        ) from exc

    except ValueError as exc:
        raise HTTPException(
            status_code=502,
            detail="Ollama Cloud returned invalid JSON.",
        ) from exc

    message = data.get("message")

    if not isinstance(message, dict):
        raise HTTPException(
            status_code=502,
            detail="Ollama Cloud returned an invalid chat response.",
        )

    return message


# ==================================================
# SAFE ANALYSIS TOOL DISPATCHER
# ==================================================

def execute_agent_tool(
    tool_name: str,
    mode: str,
    analysis: dict,
) -> dict:
    """Execute only explicitly approved analysis tools."""

    approved_tools = {
        "inspect_analysis_evidence": inspect_analysis_evidence,
        "check_analysis_uncertainty": check_analysis_uncertainty,
    }

    tool = approved_tools.get(tool_name)

    if tool is None:
        raise ValueError(
            f"Unapproved agent tool: {tool_name}"
        )

    return tool(mode, analysis)


# ==================================================
# TOOL-USING ANALYSIS AGENT
# ==================================================

@app.post("/api/agent/explain")
def explain_analysis(
    request: AgentExplainRequest,
):
    mode = request.mode
    analysis = request.analysis

    evidence = inspect_analysis_evidence(
        mode,
        analysis,
    )

    messages = [
        {
            "role": "system",
            "content": (
                "You are Nextora's research-assistant agent. "
                "You MUST call both inspect_analysis_evidence and "
                "check_analysis_uncertainty before your final report. "
                "Use their actual returned results. Never invent findings, "
                "counts, or probabilities. Do not diagnose patients or "
                "prescribe medication. Do not describe model confidence "
                "as a calibrated probability of disease. In single-cell "
                "mode, discuss only the isolated-cell result. In "
                "whole-smear mode, explain that candidate regions are "
                "heuristic and are not necessarily confirmed individual "
                "cells. Do not call candidate counts a validated parasite "
                "count or parasitemia. The model has not been clinically "
                "validated. Finish with limitations and qualified "
                "laboratory confirmation."
            ),
        },
        {
            "role": "user",
            "content": (
                f"Analyze this Nextora {mode} result. Call both required "
                "Python tools, examine their results, and produce a concise "
                "evidence-based report.\n\n"
                "Supplied analysis:\n"
                + json.dumps(analysis, ensure_ascii=False)
            ),
        },
    ]

    required_tools = {
        "inspect_analysis_evidence",
        "check_analysis_uncertainty",
    }

    tools_executed = []
    tool_results = {}

    try:
        for _ in range(6):
            message = ask_ollama(messages)
            tool_calls = message.get("tool_calls", [])

            if not tool_calls:
                answer = message.get("content", "").strip()

                if not answer:
                    raise HTTPException(
                        status_code=502,
                        detail="Ollama Cloud returned an empty response.",
                    )

                missing_tools = (
                    required_tools - set(tools_executed)
                )

                if missing_tools:
                    messages.append(message)
                    messages.append({
                        "role": "user",
                        "content": (
                            "Before answering, call the required tools "
                            "that have not yet executed: "
                            + ", ".join(sorted(missing_tools))
                        ),
                    })
                    continue

                return {
                    "mode": mode,
                    "explanation": answer,
                    "evidence": evidence,
                    "tool_results": tool_results,
                    "tools_executed": tools_executed,
                    "agent_model": OLLAMA_MODEL,
                    "agent_runtime": "Ollama Cloud",
                    "clinical_validation": False,
                    "disclaimer": (
                        "Research prototype only. This explanation is "
                        "not a diagnosis. Seek qualified laboratory "
                        "confirmation."
                    ),
                }

            messages.append(message)

            for call in tool_calls:
                function_info = call.get("function", {})
                tool_name = function_info.get("name")

                if tool_name not in required_tools:
                    raise HTTPException(
                        status_code=502,
                        detail="The agent requested an unapproved tool.",
                    )

                if tool_name not in tools_executed:
                    result = execute_agent_tool(
                        tool_name=tool_name,
                        mode=mode,
                        analysis=analysis,
                    )

                    tool_results[tool_name] = result
                    tools_executed.append(tool_name)
                else:
                    result = tool_results[tool_name]

                messages.append({
                    "role": "tool",
                    "tool_name": tool_name,
                    "content": json.dumps(
                        result,
                        ensure_ascii=False,
                    ),
                })

        raise HTTPException(
            status_code=502,
            detail=(
                "The agent could not complete its required tool calls "
                "and final report within the allowed rounds."
            ),
        )

    except HTTPException:
        raise

    except Exception as exc:
        print(
            f"Analysis agent error: {type(exc).__name__}: {exc}"
        )

        raise HTTPException(
            status_code=500,
            detail="The analysis agent failed. Check backend logs.",
        ) from exc


# ==================================================
# DETERMINISTIC TREATMENT DECISION SUPPORT
# ==================================================

@app.post("/api/treatment/recommend")
def recommend_treatment(
    request: TreatmentRequest,
):
    """
    Preliminary protocol-aware treatment decision support.
    Not a prescription or clinically validated system.
    """

    try:
        return assess_treatment(request)

    except Exception as exc:
        print(
            f"Treatment assessment error: "
            f"{type(exc).__name__}: {exc}"
        )

        raise HTTPException(
            status_code=500,
            detail="Treatment assessment failed.",
        ) from exc


# ==================================================
# TREATMENT DECISION-SUPPORT AGENT
# ==================================================

TREATMENT_AGENT_TOOLS = [
    {
        "type": "function",
        "function": {
            "name": "assess_treatment",
            "description": (
                "Run Nextora's deterministic treatment assessment "
                "using the clinical information submitted with this "
                "request. Never invent or change patient details. "
                "Do not prescribe or invent doses."
            ),
            "parameters": {
                "type": "object",
                "properties": {},
                "required": [],
            },
        },
    },
]


@app.post("/api/agent/treatment-report")
def treatment_agent_report(
    request: TreatmentRequest,
):
    """
    Uses the deterministic treatment engine and Ollama Cloud
    to explain the assessment. Not a diagnosis or prescription.
    """

    try:
        assessment = assess_treatment(request)

    except Exception as exc:
        print(
            f"Treatment assessment error: "
            f"{type(exc).__name__}: {exc}"
        )

        raise HTTPException(
            status_code=500,
            detail="Treatment assessment failed.",
        ) from exc

    if hasattr(request, "model_dump"):
        submitted_data = request.model_dump()
    else:
        submitted_data = request.dict()

    messages = [
        {
            "role": "system",
            "content": (
                "You are Nextora's treatment decision-support "
                "explanation agent. You must call the approved "
                "assess_treatment tool before responding. Use only "
                "the tool's returned assessment. Never invent a drug, "
                "dose, frequency, duration, contraindication, laboratory "
                "finding, or patient detail. Do not override the "
                "assessment's status or warnings. If the result says "
                "protocol_review_required, explain that the regimen "
                "requires verification against the applicable Nigerian "
                "protocol by a qualified clinician. If dose is null, "
                "explicitly state that no dose was calculated. If the "
                "result indicates emergency escalation, emphasize urgent "
                "hospital assessment. Do not present this output as a "
                "prescription. The system cannot independently verify "
                "that laboratory confirmation occurred. State that this "
                "prototype has not been clinically validated."
            ),
        },
        {
            "role": "user",
            "content": (
                "Call the approved treatment assessment tool and explain "
                "its result, urgency, unverified information, and next "
                "step. Do not create a prescription.\n\n"
                "Submitted clinical information:\n"
                + json.dumps(
                    submitted_data,
                    ensure_ascii=False,
                )
            ),
        },
    ]

    tools_executed = False

    try:
        for _ in range(4):
            message = ask_ollama(
                messages,
                tools=TREATMENT_AGENT_TOOLS,
            )

            tool_calls = message.get("tool_calls", [])

            if not tool_calls:
                explanation = message.get("content", "").strip()

                if not explanation:
                    raise HTTPException(
                        status_code=502,
                        detail=(
                            "Ollama Cloud returned an empty response."
                        ),
                    )

                if not tools_executed:
                    messages.append(message)
                    messages.append({
                        "role": "user",
                        "content": (
                            "You must call the approved "
                            "assess_treatment tool before explaining "
                            "the result."
                        ),
                    })
                    continue

                return {
                    "assessment": assessment,
                    "explanation": explanation,
                    "tools_executed": ["assess_treatment"],
                    "agent_model": OLLAMA_MODEL,
                    "agent_runtime": "Ollama Cloud",
                    "clinical_validation": False,
                    "disclaimer": (
                        "Research prototype only. This output is not "
                        "a diagnosis or prescription. A qualified "
                        "clinician must assess the patient and verify "
                        "the applicable treatment protocol."
                    ),
                }

            messages.append(message)

            for call in tool_calls:
                function_info = call.get("function", {})
                tool_name = function_info.get("name")

                if tool_name != "assess_treatment":
                    raise HTTPException(
                        status_code=502,
                        detail="The agent requested an unapproved tool.",
                    )

                # The model cannot supply or change clinical parameters.
                messages.append({
                    "role": "tool",
                    "tool_name": "assess_treatment",
                    "content": json.dumps(
                        assessment,
                        ensure_ascii=False,
                    ),
                })

                tools_executed = True

        raise HTTPException(
            status_code=502,
            detail=(
                "The treatment agent could not complete its required "
                "tool call and explanation."
            ),
        )

    except HTTPException:
        raise

    except Exception as exc:
        print(
            f"Treatment agent error: {type(exc).__name__}: {exc}"
        )

        raise HTTPException(
            status_code=500,
            detail="The treatment agent failed. Check backend logs.",
        ) from exc