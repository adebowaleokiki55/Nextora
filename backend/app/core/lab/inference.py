"""PlasmoSENet model loading, image validation and single-cell classification.

Moved unchanged in behaviour from the original backend/main.py.
"""

from __future__ import annotations

import gc
import hashlib
import threading
from io import BytesIO
from pathlib import Path

import cv2
import torch
from PIL import Image, UnidentifiedImageError
from torchvision import transforms

from backend.app.core.lab.plasmosenet import PlasmoSENet

CLASS_NAMES = {
    0: "Parasitized (Infected)",
    1: "Uninfected",
}

MODEL_NAME = "PlasmoSENet"
MAX_UPLOAD_BYTES = 10 * 1024 * 1024
MAX_IMAGE_PIXELS = 12_000_000
ALLOWED_CONTENT_TYPES = {"image/jpeg", "image/png", "image/webp"}

# Render's free instance is CPU-based and memory-constrained.
DEVICE = torch.device("cpu")

torch.set_num_threads(1)
try:
    torch.set_num_interop_threads(1)
except RuntimeError:
    pass
cv2.setNumThreads(0)

TRANSFORM = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225],
    ),
])


class ImageInputError(Exception):
    """A rejected upload, carrying an HTTP-style status for the API layer."""

    def __init__(self, status_code: int, detail: str):
        super().__init__(detail)
        self.status_code = status_code
        self.detail = detail


# --------------------------------------------------------------------------
# Model
# --------------------------------------------------------------------------

_model: PlasmoSENet | None = None
_model_lock = threading.Lock()


def load_model(model_path: Path) -> PlasmoSENet:
    if not model_path.exists():
        raise RuntimeError(f"Model weights not found: {model_path}")

    loaded_model = PlasmoSENet(num_classes=2)
    state_dict = torch.load(model_path, map_location="cpu", weights_only=True)
    loaded_model.load_state_dict(state_dict)
    loaded_model.to(DEVICE)
    loaded_model.eval()

    del state_dict
    gc.collect()
    return loaded_model


def get_model(model_path: Path) -> PlasmoSENet:
    """Load the model once per process and reuse it."""
    global _model
    if _model is None:
        with _model_lock:
            if _model is None:
                _model = load_model(model_path)
    return _model


def is_model_loaded() -> bool:
    return _model is not None


_model_versions: dict[Path, str] = {}


def get_model_version(model_path: Path) -> str:
    """Short SHA-256 of the weights file, stored with every lab result."""
    if model_path not in _model_versions:
        digest = hashlib.sha256()
        with open(model_path, "rb") as weights:
            for chunk in iter(lambda: weights.read(1 << 20), b""):
                digest.update(chunk)
        _model_versions[model_path] = f"sha256:{digest.hexdigest()[:12]}"
    return _model_versions[model_path]


# --------------------------------------------------------------------------
# Upload validation
# --------------------------------------------------------------------------

def validate_image_dimensions(image_bytes: bytes) -> None:
    try:
        with Image.open(BytesIO(image_bytes)) as image:
            width, height = image.size

            if width <= 0 or height <= 0:
                raise ImageInputError(400, "The image dimensions are invalid.")

            if width * height > MAX_IMAGE_PIXELS:
                raise ImageInputError(
                    413,
                    "The image dimensions are too large. "
                    "Please upload an image containing no more "
                    "than 12 million pixels.",
                )

            image.verify()

    except ImageInputError:
        raise

    except (UnidentifiedImageError, OSError, ValueError) as exc:
        raise ImageInputError(
            400, "The uploaded file could not be processed as an image."
        ) from exc


# --------------------------------------------------------------------------
# Single-cell classification
# --------------------------------------------------------------------------

def classify_cell_image(image_bytes: bytes, model: PlasmoSENet) -> dict:
    """Classify one isolated cell image. Raises ImageInputError for bad images."""
    image = None
    input_tensor = None
    logits = None
    probabilities = None

    try:
        with Image.open(BytesIO(image_bytes)) as source:
            image = source.convert("RGB")

        input_tensor = TRANSFORM(image).unsqueeze(0).to(DEVICE)

        with torch.inference_mode():
            logits = model(input_tensor)
            probabilities = torch.softmax(logits, dim=1)[0]

            if probabilities.numel() != 2:
                raise RuntimeError("Unexpected model output shape.")

            predicted_class = int(torch.argmax(probabilities).item())
            confidence = float(probabilities[predicted_class].item())
            infected_probability = float(probabilities[0].item())
            uninfected_probability = float(probabilities[1].item())

        return {
            "predicted_class": predicted_class,
            "prediction": CLASS_NAMES[predicted_class],
            "confidence": round(confidence * 100, 2),
            "probabilities": {
                "parasitized": round(infected_probability * 100, 2),
                "uninfected": round(uninfected_probability * 100, 2),
            },
            "model": MODEL_NAME,
            "disclaimer": (
                "Research screening result for an isolated cell image; "
                "not a clinical diagnosis."
            ),
        }

    except (UnidentifiedImageError, OSError, ValueError) as exc:
        raise ImageInputError(
            400, "The uploaded file could not be processed as an image."
        ) from exc

    finally:
        image = None
        input_tensor = None
        logits = None
        probabilities = None
        gc.collect()
