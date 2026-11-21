import numpy as np
import cv2
import pytest

from backend.app.core.lab.smear_analysis import detect_candidate_cells


def draw_smear(centres, radius=30, size=(600, 800), seed=1):
    img = np.full((*size, 3), (232, 226, 238), np.uint8)
    for centre in centres:
        cv2.circle(img, centre, radius, (182, 140, 205), -1)
    for centre in centres:
        cv2.circle(img, centre, radius, (125, 85, 155), 2)
    noise = np.random.default_rng(seed).normal(0, 4, img.shape)
    return np.clip(img + noise, 0, 255).astype(np.uint8)


def boxes_contain(boxes, centres):
    """Every true centre lies inside exactly one detected box."""
    for cx, cy in centres:
        hits = [b for b in boxes if b[0] <= cx <= b[0] + b[2] and b[1] <= cy <= b[1] + b[3]]
        if len(hits) != 1:
            return False
    return True


def test_isolated_cells():
    centres = [(100, 100), (300, 100), (500, 100), (100, 300), (300, 300)]
    boxes = detect_candidate_cells(draw_smear(centres))
    assert len(boxes) == 5
    assert boxes_contain(boxes, centres)


@pytest.mark.parametrize("radius", [18, 30, 45])
@pytest.mark.parametrize("gap", [1.0, 0.85, 0.7])
def test_touching_and_overlapping_pairs_are_split(radius, gap):
    step = int(2 * radius * gap)
    centres = []
    for row in range(3):
        x, y = 4 * radius, 3 * radius + row * 5 * radius
        centres += [(x, y), (x + step, y)]
    boxes = detect_candidate_cells(draw_smear(centres, radius, size=(20 * radius, 12 * radius)))
    assert len(boxes) == len(centres)
    assert boxes_contain(boxes, centres)


def test_clump_of_three():
    centres = [(300, 300), (360, 300), (330, 350), (600, 150)]
    boxes = detect_candidate_cells(draw_smear(centres))
    assert len(boxes) == 4
    assert boxes_contain(boxes, centres)


def test_debris_is_ignored():
    img = draw_smear([(200, 200)])
    cv2.circle(img, (500, 400), 6, (120, 80, 150), -1)  # speck, far below min area
    assert len(detect_candidate_cells(img)) == 1


def test_large_image_is_scaled_back_to_original_coordinates():
    centres = [(400, 400), (1200, 900), (2600, 1600)]
    img = draw_smear(centres, radius=60, size=(2000, 3000))
    boxes = detect_candidate_cells(img)
    assert len(boxes) == 3
    assert boxes_contain(boxes, centres)


def test_empty_image_rejected():
    with pytest.raises(ValueError):
        detect_candidate_cells(np.zeros((0, 0, 3), np.uint8))
