
from pathlib import Path

import cv2
import numpy as np
import torch
from PIL import Image
from torchvision import transforms


CLASS_NAMES = {
    0: "Parasitized (Infected)",
    1: "Uninfected",
}

TRANSFORM = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225],
    ),
])


def detect_candidate_cells(
    image: np.ndarray,
    min_area: int = 500,
    max_area: int = 50000,
):
    """
    Find candidate cell regions using contours.

    These are heuristic candidates, not guaranteed cell detections.
    Parameters may need adjustment for the microscopy image.
    """
    if image is None or image.size == 0:
        raise ValueError("The supplied image is empty.")

    height, width = image.shape[:2]

 
    scale = min(1.0, 1600 / max(height, width))
    if scale < 1.0:
        working = cv2.resize(
            image,
            (int(width * scale), int(height * scale)),
            interpolation=cv2.INTER_AREA,
        )
    else:
        working = image.copy()

    gray = cv2.cvtColor(working, cv2.COLOR_BGR2GRAY)
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)

   
    thresh = cv2.adaptiveThreshold(
        blurred,
        255,
        cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
        cv2.THRESH_BINARY_INV,
        51,
        7,
    )

   
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
    cleaned = cv2.morphologyEx(
        thresh,
        cv2.MORPH_OPEN,
        kernel,
        iterations=1,
    )

    contours, _ = cv2.findContours(
        cleaned,
        cv2.RETR_EXTERNAL,
        cv2.CHAIN_APPROX_SIMPLE,
    )

    candidates = []

    for contour in contours:
        area = cv2.contourArea(contour)
        if not min_area <= area <= max_area:
            continue

        x, y, w, h = cv2.boundingRect(contour)

        
        aspect_ratio = w / max(h, 1)
        if not 0.55 <= aspect_ratio <= 1.8:
            continue

        
        if scale < 1.0:
            x = int(x / scale)
            y = int(y / scale)
            w = int(w / scale)
            h = int(h / scale)

        candidates.append((x, y, w, h))

    
    candidates.sort(key=lambda box: (box[1], box[0]))
    return candidates


def analyze_smear(
    image_bytes: bytes,
    model,
    device: torch.device,
    min_area: int = 500,
    max_area: int = 50000,
    batch_size: int = 16,
):
    """
    Analyse a whole microscopy image using heuristic cell candidates
    and the existing PlasmoSENet classifier.

    Returns predictions and an annotated PNG image.
    """
    if not image_bytes:
        raise ValueError("The image is empty.")

    image_array = np.frombuffer(image_bytes, dtype=np.uint8)
    image = cv2.imdecode(image_array, cv2.IMREAD_COLOR)

    if image is None:
        raise ValueError("Could not decode the supplied image.")

    height, width = image.shape[:2]
    candidates = detect_candidate_cells(
        image,
        min_area=min_area,
        max_area=max_area,
    )

    annotated = image.copy()
    results = []
    crops = []

    
    for x, y, w, h in candidates:
        margin = max(4, int(min(w, h) * 0.08))
        x1 = max(0, x - margin)
        y1 = max(0, y - margin)
        x2 = min(width, x + w + margin)
        y2 = min(height, y + h + margin)

        crop = image[y1:y2, x1:x2]
        if crop.size == 0:
            continue

        rgb = cv2.cvtColor(crop, cv2.COLOR_BGR2RGB)
        crops.append((TRANSFORM(Image.fromarray(rgb)), (x, y, w, h)))

    
    for start in range(0, len(crops), batch_size):
        batch = crops[start:start + batch_size]
        tensors = torch.stack([item[0] for item in batch]).to(device)

        with torch.inference_mode():
            probabilities = torch.softmax(model(tensors), dim=1)

        for (_, (x, y, w, h)), probs in zip(batch, probabilities):
            predicted_class = int(torch.argmax(probs).item())
            confidence = float(probs[predicted_class].item()) * 100

            result = {
                "cell_id": len(results) + 1,
                "box": {
                    "x": x,
                    "y": y,
                    "width": w,
                    "height": h,
                },
                "predicted_class": predicted_class,
                "prediction": CLASS_NAMES[predicted_class],
                "confidence": round(confidence, 2),
            }
            results.append(result)

           
            infected = predicted_class == 0
            color = (0, 0, 255) if infected else (0, 180, 0)
            cv2.rectangle(
                annotated,
                (x, y),
                (x + w, y + h),
                color,
                2,
            )

            label = (
                f"#{result['cell_id']} "
                f"{'Infected' if infected else 'Uninfected'} "
                f"{confidence:.0f}%"
            )

            text_y = max(18, y - 7)
            cv2.putText(
                annotated,
                label,
                (x, text_y),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.45,
                color,
                1,
                cv2.LINE_AA,
            )

    infected_count = sum(
        result["predicted_class"] == 0 for result in results
    )
    uninfected_count = sum(
        result["predicted_class"] == 1 for result in results
    )

    success, encoded = cv2.imencode(".png", annotated)
    if not success:
        raise RuntimeError("Could not encode the annotated image.")

    return {
        "image_width": width,
        "image_height": height,
        "candidate_count": len(results),
        "candidate_infected_count": infected_count,
        "candidate_uninfected_count": uninfected_count,
        "cells": results,
        "annotated_image": encoded.tobytes(),
        "disclaimer": (
            "Experimental image analysis. Contours may not correspond to "
            "individual cells. Results are not a clinical diagnosis or a "
            "validated parasitemia measurement."
        ),
    }
