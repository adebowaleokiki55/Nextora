
import gc

import cv2
import numpy as np
import torch
from PIL import Image
from torchvision import transforms


CLASS_NAMES = {
    0: "Parasitized (Infected)",
    1: "Uninfected",
}

MAX_WORKING_SIDE = 1600
MAX_ANNOTATION_SIDE = 1400
MAX_BATCH_SIZE = 4

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
    Find heuristic candidate regions using contours.

    These candidates are not guaranteed to be individual cells.
    Contour area thresholds apply to the working image.
    """
    if image is None or image.size == 0:
        raise ValueError("The supplied image is empty.")

    height, width = image.shape[:2]

    scale = min(
        1.0,
        MAX_WORKING_SIDE / max(height, width),
    )

    if scale < 1.0:
        working = cv2.resize(
            image,
            (
                max(1, int(width * scale)),
                max(1, int(height * scale)),
            ),
            interpolation=cv2.INTER_AREA,
        )
    else:
        working = image.copy()

    gray = None
    blurred = None
    thresh = None
    cleaned = None
    contours = None
    kernel = None

    try:
        gray = cv2.cvtColor(working, cv2.COLOR_BGR2GRAY)

        blurred = cv2.GaussianBlur(
            gray,
            (5, 5),
            0,
        )

        thresh = cv2.adaptiveThreshold(
            blurred,
            255,
            cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
            cv2.THRESH_BINARY_INV,
            51,
            7,
        )

        kernel = cv2.getStructuringElement(
            cv2.MORPH_ELLIPSE,
            (3, 3),
        )

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

            # Clamp boxes to the original image boundaries.
            x = max(0, min(x, width - 1))
            y = max(0, min(y, height - 1))
            w = max(1, min(w, width - x))
            h = max(1, min(h, height - y))

            candidates.append((x, y, w, h))

        candidates.sort(
            key=lambda box: (box[1], box[0])
        )

        return candidates

    finally:
        # Release the temporary contour-processing arrays.
        del working
        gray = None
        blurred = None
        thresh = None
        cleaned = None
        contours = None
        kernel = None


def analyze_smear(
    image_bytes: bytes,
    model,
    device: torch.device,
    min_area: int = 500,
    max_area: int = 50000,
    batch_size: int = MAX_BATCH_SIZE,
):
    """
    Analyse a microscopy image using heuristic candidate regions
    and the existing PlasmoSENet classifier.

    Memory optimizations:
    - Processes crops in small batches instead of storing every crop.
    - Limits inference batch size.
    - Uses a reduced-size annotation preview.
    - Releases temporary tensors and arrays between batches.

    Returns predictions and a PNG-encoded annotation preview.
    """
    if not image_bytes:
        raise ValueError("The image is empty.")

    image_array = np.frombuffer(
        image_bytes,
        dtype=np.uint8,
    )

    image = cv2.imdecode(
        image_array,
        cv2.IMREAD_COLOR,
    )

    del image_array

    if image is None:
        raise ValueError("Could not decode the supplied image.")

    height, width = image.shape[:2]

    candidates = detect_candidate_cells(
        image,
        min_area=min_area,
        max_area=max_area,
    )

    # Create a smaller annotation canvas to avoid copying the
    # entire original-resolution image a second time.
    annotation_scale = min(
        1.0,
        MAX_ANNOTATION_SIDE / max(height, width),
    )

    if annotation_scale < 1.0:
        annotated = cv2.resize(
            image,
            (
                max(1, int(width * annotation_scale)),
                max(1, int(height * annotation_scale)),
            ),
            interpolation=cv2.INTER_AREA,
        )
    else:
        annotated = image.copy()

    annotation_height, annotation_width = annotated.shape[:2]

    # Clamp requested batch size to a conservative limit.
    try:
        batch_size = int(batch_size)
    except (TypeError, ValueError):
        batch_size = MAX_BATCH_SIZE

    batch_size = max(
        1,
        min(batch_size, MAX_BATCH_SIZE),
    )

    results = []

    try:
        for start in range(0, len(candidates), batch_size):
            candidate_batch = candidates[
                start:start + batch_size
            ]

            tensors = []
            valid_boxes = []

            # Only keep crops for the current small batch.
            for x, y, w, h in candidate_batch:
                margin = max(
                    4,
                    int(min(w, h) * 0.08),
                )

                x1 = max(0, x - margin)
                y1 = max(0, y - margin)
                x2 = min(width, x + w + margin)
                y2 = min(height, y + h + margin)

                crop = image[y1:y2, x1:x2]

                if crop.size == 0:
                    continue

                rgb = cv2.cvtColor(
                    crop,
                    cv2.COLOR_BGR2RGB,
                )

                try:
                    pil_image = Image.fromarray(rgb)
                    tensor = TRANSFORM(pil_image)
                    tensors.append(tensor)
                    valid_boxes.append((x, y, w, h))
                finally:
                    del crop
                    del rgb

                    if "pil_image" in locals():
                        del pil_image

            if not tensors:
                continue

            batch_tensor = None
            logits = None
            probabilities = None

            try:
                batch_tensor = torch.stack(
                    tensors,
                    dim=0,
                ).to(device)

                with torch.inference_mode():
                    logits = model(batch_tensor)
                    probabilities = torch.softmax(
                        logits,
                        dim=1,
                    )

                    if (
                        probabilities.ndim != 2
                        or probabilities.shape[1] != 2
                    ):
                        raise RuntimeError(
                            "Unexpected model output shape."
                        )

                    predicted_classes = (
                        torch.argmax(
                            probabilities,
                            dim=1,
                        )
                        .cpu()
                        .tolist()
                    )

                    confidence_values = (
                        probabilities.max(dim=1)
                        .values
                        .cpu()
                        .tolist()
                    )

                for (
                    (x, y, w, h),
                    predicted_class,
                    confidence_value,
                ) in zip(
                    valid_boxes,
                    predicted_classes,
                    confidence_values,
                ):
                    confidence = float(confidence_value) * 100

                    cell_id = len(results) + 1
                    infected = predicted_class == 0

                    result = {
                        "cell_id": cell_id,
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

                    # Convert original-image coordinates to preview
                    # coordinates before drawing the annotation.
                    ax1 = int(x * annotation_scale)
                    ay1 = int(y * annotation_scale)
                    ax2 = int((x + w) * annotation_scale)
                    ay2 = int((y + h) * annotation_scale)

                    color = (
                        (0, 0, 255)
                        if infected
                        else (0, 180, 0)
                    )

                    cv2.rectangle(
                        annotated,
                        (ax1, ay1),
                        (ax2, ay2),
                        color,
                        2,
                    )

                    label = (
                        f"#{cell_id} "
                        f"{'Infected' if infected else 'Uninfected'} "
                        f"{confidence:.0f}%"
                    )

                    text_y = max(18, ay1 - 7)

                    cv2.putText(
                        annotated,
                        label,
                        (ax1, text_y),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.45,
                        color,
                        1,
                        cv2.LINE_AA,
                    )

            finally:
                # Drop tensor references before processing the next batch.
                del tensors
                del valid_boxes

                if batch_tensor is not None:
                    del batch_tensor

                if logits is not None:
                    del logits

                if probabilities is not None:
                    del probabilities

                gc.collect()

        infected_count = sum(
            result["predicted_class"] == 0
            for result in results
        )

        uninfected_count = sum(
            result["predicted_class"] == 1
            for result in results
        )

        success, encoded = cv2.imencode(
            ".png",
            annotated,
            [cv2.IMWRITE_PNG_COMPRESSION, 3],
        )

        if not success:
            raise RuntimeError(
                "Could not encode the annotated image."
            )

        annotated_bytes = encoded.tobytes()

        del encoded

        return {
            "image_width": width,
            "image_height": height,
            "annotation_width": annotation_width,
            "annotation_height": annotation_height,
            "candidate_count": len(results),
            "candidate_infected_count": infected_count,
            "candidate_uninfected_count": uninfected_count,
            "cells": results,
            "annotated_image": annotated_bytes,
            "disclaimer": (
                "Experimental image analysis. Contours may not "
                "correspond to individual cells. Candidate counts "
                "and classifications are not a clinical diagnosis "
                "or a validated parasitemia measurement."
            ),
        }

    finally:
        # Release large image buffers when analysis ends.
        candidates.clear()
        del annotated
        del image
        gc.collect()
