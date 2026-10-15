
from io import BytesIO
from pathlib import Path

import torch
from PIL import Image, UnidentifiedImageError
from torchvision import transforms
from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware

from plasmosenet import PlasmoSENet

# Resolve paths from this file, not the terminal's working directory.
PROJECT_ROOT = Path(__file__).resolve().parent.parent
MODEL_PATH = PROJECT_ROOT / "model.pth"

MAX_UPLOAD_BYTES = 10 * 1024 * 1024  # 10 MB

CLASS_NAMES = {
    0: "Parasitized (Infected)",
    1: "Uninfected",
}

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

app = FastAPI(
    title="MalariaLens AI API",
    version="1.0.0",
    description="Research prototype for malaria blood-cell image classification.",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "http://localhost:5173",
        "http://127.0.0.1:5173",
        "https://nextora-a5yj6qcjy-adebowale-okikis-projects.vercel.app/"
    ],
    allow_credentials=False,
    allow_methods=["GET", "POST"],
    allow_headers=["*"],
)

# Match the preprocessing used by your verified inference script.
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225],
    ),
])


def load_model():
    if not MODEL_PATH.exists():
        raise RuntimeError(f"Model weights not found: {MODEL_PATH}")

    model = PlasmoSENet(num_classes=2)

    # weights_only=True is appropriate for a trusted state_dict file
    # on current PyTorch versions.
    state_dict = torch.load(
        MODEL_PATH,
        map_location=DEVICE,
        weights_only=True,
    )

    model.load_state_dict(state_dict)
    model.to(DEVICE)
    model.eval()

    return model


# Loading at module startup means we do not reload weights per request.
model = load_model()


@app.get("/api/health")
def health():
    return {
        "status": "ok",
        "model_loaded": model is not None,
        "device": str(DEVICE),
    }


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

    try:
        # Read at most the configured limit plus one byte.
        image_bytes = file.file.read(MAX_UPLOAD_BYTES + 1)

        if not image_bytes:
            raise HTTPException(status_code=400, detail="The file is empty.")

        if len(image_bytes) > MAX_UPLOAD_BYTES:
            raise HTTPException(
                status_code=413,
                detail="Image exceeds the 10 MB upload limit.",
            )

        # Verify that the file is a decodable image, not just a renamed file.
        with Image.open(BytesIO(image_bytes)) as check_image:
            check_image.verify()

        with Image.open(BytesIO(image_bytes)) as source:
            image = source.convert("RGB")

        input_tensor = transform(image).unsqueeze(0).to(DEVICE)

        with torch.inference_mode():
            logits = model(input_tensor)
            probabilities = torch.softmax(logits, dim=1)[0]

        if probabilities.numel() != 2:
            raise RuntimeError("Model returned an unexpected output shape.")

        predicted_class = int(torch.argmax(probabilities).item())
        confidence = float(probabilities[predicted_class].item())

        return {
            "predicted_class": predicted_class,
            "prediction": CLASS_NAMES[predicted_class],
            "confidence": round(confidence * 100, 2),
            "probabilities": {
                "parasitized": round(float(probabilities[0]) * 100, 2),
                "uninfected": round(float(probabilities[1]) * 100, 2),
            },
            "model": "PlasmoSENet",
            "disclaimer": (
                "Research screening result for an isolated cell image; "
                "not a clinical diagnosis."
            ),
        }

    except HTTPException:
        raise
    except (UnidentifiedImageError, OSError, ValueError):
        raise HTTPException(
            status_code=400,
            detail="The uploaded file could not be processed as an image.",
        )
    except Exception:
        # Log the full exception privately in production.
        raise HTTPException(
            status_code=500,
            detail="Prediction failed. Check the backend logs.",
        )
