# -*- coding: utf-8 -*-
"""
Created on Tue Aug 18 13:52:51 2026

@author: Kim Bjerge
"""

from pathlib import Path
import cv2
import numpy as np

from sahi import AutoDetectionModel
from sahi.predict import get_sliced_prediction

from sklearn.metrics import precision_recall_fscore_support


# ============================================================
# Configuration
# ============================================================

#MODEL_PATH = "F:/insectsDCT/runs/detect/insects7Color/weights/best.pt"
#IMAGE_DIR = Path("D:\Odin\ANIMK\images")
#LABEL_DIR = Path("D:\Odin\ANIMK\images")

MODEL_PATH = "/home/don/insectsDCT/runs/detect/insects7Color/weights/best.pt"

# Directory containing the test images
IMAGE_DIR = Path("/home/don/insectsDCT/datasets/insectsTest/testOdin/images")

# Directory containing YOLO ground-truth labels
LABEL_DIR = Path("/home/don/insectsDCT/datasets/insectsTest/testOdin/labels")

# Results
TEST_NAME = "insects7Color_SAHI"

# SAHI parameters
#SLICE_HEIGHT = 1280
#SLICE_WIDTH = 1280
SLICE_HEIGHT = 1080
SLICE_WIDTH = 1920

OVERLAP_HEIGHT = 0.20
OVERLAP_WIDTH = 0.20

# YOLO confidence threshold
CONFIDENCE = 0.25

# Number of slices processed together
BATCH_SIZE = 8

# GPU
DEVICE = "cuda:0"

# IoU used for matching predictions to ground truth
#MATCH_IOU = 0.50
MATCH_IOU = 0.10


# ============================================================
# Load YOLO model through SAHI
# ============================================================

detection_model = AutoDetectionModel.from_pretrained(
    model_type="ultralytics",
    model_path=MODEL_PATH,
    confidence_threshold=CONFIDENCE,
    device=DEVICE,
)


# ============================================================
# Utility functions
# ============================================================

def load_yolo_labels(label_file, image_width, image_height):

    """
    Read YOLO labels:

        class x_center y_center width height

    and convert them to:

        class x1 y1 x2 y2
    """

    ground_truth = []

    if not label_file.exists():
        return ground_truth

    with open(label_file, "r") as f:

        for line in f:

            values = line.strip().split()

            if len(values) != 5:
                continue

            class_id = int(values[0])

            x_center = float(values[1]) * image_width
            y_center = float(values[2]) * image_height
            width = float(values[3]) * image_width
            height = float(values[4]) * image_height

            x1 = x_center - width / 2
            y1 = y_center - height / 2
            x2 = x_center + width / 2
            y2 = y_center + height / 2

            ground_truth.append({
                "class_id": class_id,
                "bbox": [x1, y1, x2, y2]
            })

    return ground_truth


def calculate_iou(box1, box2):

    """
    Calculate IoU between:

        [x1, y1, x2, y2]
    """

    x1 = max(box1[0], box2[0])
    y1 = max(box1[1], box2[1])

    x2 = min(box1[2], box2[2])
    y2 = min(box1[3], box2[3])

    intersection_width = max(0, x2 - x1)
    intersection_height = max(0, y2 - y1)

    intersection = intersection_width * intersection_height

    area1 = max(0, box1[2] - box1[0]) * \
            max(0, box1[3] - box1[1])

    area2 = max(0, box2[2] - box2[0]) * \
            max(0, box2[3] - box2[1])

    union = area1 + area2 - intersection

    if union == 0:
        return 0

    return intersection / union


# ============================================================
# Evaluate one image
# ============================================================

def evaluate_image(image_path):

    # --------------------------------------------------------
    # Read image
    # --------------------------------------------------------

    image = cv2.imread(str(image_path))

    if image is None:
        print(f"Could not read: {image_path}")
        return 0, 0, 0

    height, width = image.shape[:2]

    # --------------------------------------------------------
    # Ground truth
    # --------------------------------------------------------

    label_path = LABEL_DIR / f"{image_path.stem}.txt"

    ground_truth = load_yolo_labels(
        label_path,
        width,
        height
    )

    # --------------------------------------------------------
    # SAHI prediction
    # --------------------------------------------------------

    result = get_sliced_prediction(
        str(image_path),
        detection_model,

        slice_height=SLICE_HEIGHT,
        slice_width=SLICE_WIDTH,

        overlap_height_ratio=OVERLAP_HEIGHT,
        overlap_width_ratio=OVERLAP_WIDTH,

        batch_size=BATCH_SIZE,

        verbose=0,
    )

    predictions = []

    for pred in result.object_prediction_list:

        bbox = pred.bbox.to_xyxy()

        predictions.append({
            "class_id": int(pred.category.id),
            "bbox": bbox,
            "score": float(pred.score.value)
        })

    # --------------------------------------------------------
    # Match predictions to ground truth
    # --------------------------------------------------------

    matched_gt = set()

    true_positive = 0
    false_positive = 0
    false_negative = 0

    # Highest confidence predictions first
    predictions.sort(
        key=lambda x: x["score"],
        reverse=True
    )

    for pred in predictions:

        best_iou = 0
        best_gt = None

        for i, gt in enumerate(ground_truth):

            if i in matched_gt:
                continue

            # Classes must match
            if pred["class_id"] != gt["class_id"]:
                continue

            iou = calculate_iou(
                pred["bbox"],
                gt["bbox"]
            )

            if iou > best_iou:
                best_iou = iou
                best_gt = i

        if best_gt is not None and best_iou >= MATCH_IOU:

            true_positive += 1
            matched_gt.add(best_gt)

        else:

            false_positive += 1

    false_negative = len(ground_truth) - len(matched_gt)

    return (
        true_positive,
        false_positive,
        false_negative
    )


# ============================================================
# Run evaluation
# ============================================================

image_files = sorted(
    list(IMAGE_DIR.glob("*.jpg")) +
    list(IMAGE_DIR.glob("*.JPG")) +
    list(IMAGE_DIR.glob("*.jpeg")) +
    list(IMAGE_DIR.glob("*.png"))
)

print()
print("=" * 70)
print("SAHI YOLO Evaluation")
print("=" * 70)

print(f"Images       : {len(image_files)}")
print(f"Slice size   : {SLICE_WIDTH} x {SLICE_HEIGHT}")
print(f"Overlap      : {OVERLAP_WIDTH}")
print(f"Confidence   : {CONFIDENCE}")
print(f"Match IoU    : {MATCH_IOU}")
print("=" * 70)


total_tp = 0
total_fp = 0
total_fn = 0


for i, image_path in enumerate(image_files):

    print(
        f"[{i+1}/{len(image_files)}] "
        f"{image_path.name}"
    )

    tp, fp, fn = evaluate_image(image_path)

    total_tp += tp
    total_fp += fp
    total_fn += fn


# ============================================================
# Calculate metrics
# ============================================================

precision = (
    total_tp / (total_tp + total_fp)
    if total_tp + total_fp > 0
    else 0
)

recall = (
    total_tp / (total_tp + total_fn)
    if total_tp + total_fn > 0
    else 0
)

f1 = (
    2 * precision * recall /
    (precision + recall)
    if precision + recall > 0
    else 0
)


# ============================================================
# Print results
# ============================================================

print()
print("=" * 70)
print("RESULTS")
print("=" * 70)

print(f"TP        : {total_tp}")
print(f"FP        : {total_fp}")
print(f"FN        : {total_fn}")

print()
print(f"Precision : {precision:.4f}")
print(f"Recall    : {recall:.4f}")
print(f"F1        : {f1:.4f}")

print()
print(
    f"{TEST_NAME},"
    f"{precision:.4f},"
    f"{recall:.4f},"
    f"{f1:.4f}"
)