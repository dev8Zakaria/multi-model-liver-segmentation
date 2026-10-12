"""Run the deployed ONNX pipeline on the held-out vessel test set and print per-class Dice.

Checks that the web app's preprocessing + ONNX models reproduce the behaviour
of the training notebooks end to end.

    python tools/evaluate_test_set.py --dataset C:/.../VascularModelDataset [--limit 200]
"""
import argparse
import glob
import os
import re
import sys

import cv2
import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from app import inference  # noqa: E402

# Background is black in Training/ and white in Testing/: both map to class 0.
MATCH_COLORS = np.array(
    [[0, 0, 0], [255, 255, 255], [0, 0, 255], [0, 255, 0], [255, 0, 0], [128, 0, 128]], dtype=np.int32)
MATCH_TO_CLASS = np.array([0, 0, 1, 2, 3, 4])


def load_ground_truth(path):
    mask = cv2.cvtColor(cv2.imread(path, cv2.IMREAD_COLOR), cv2.COLOR_BGR2RGB)
    mask = cv2.resize(mask, (inference.PATCH_SIZE, inference.PATCH_SIZE), interpolation=cv2.INTER_NEAREST)
    dists = ((mask.reshape(-1, 1, 3).astype(np.int32) - MATCH_COLORS[None]) ** 2).sum(axis=2)
    return MATCH_TO_CLASS[dists.argmin(axis=1)].reshape(mask.shape[:2])


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", required=True)
    parser.add_argument("--limit", type=int, default=0, help="evaluate only N evenly spaced pairs (0 = all)")
    args = parser.parse_args()

    img_dir = os.path.join(args.dataset, "Testing", "testset_images")
    mask_dir = os.path.join(args.dataset, "Testing", "testset_masks")
    ids = sorted(int(re.search(r"(\d+)", os.path.basename(p)).group(1))
                 for p in glob.glob(os.path.join(img_dir, "patch_*.tiff")))
    ids = [i for i in ids if os.path.exists(os.path.join(mask_dir, f"mask_{i}.tiff"))]
    if args.limit:
        ids = ids[::max(1, len(ids) // args.limit)][:args.limit]
    print(f"{len(ids)} test pairs")

    n = len(inference.CLASSES)
    names = [c["key"] for c in inference.CLASSES]
    print(f"{'model':16s}" + "".join(f"{name:>14s}" for name in names) + f"{'mean(1-4)':>12s}")
    for model in inference.REGISTRY["models"]:
        confusion = np.zeros((n, n), dtype=np.int64)
        for i in ids:
            with open(os.path.join(img_dir, f"patch_{i}.tiff"), "rb") as f:
                rgb = inference.decode_image(f.read())
            pred, _ = inference.segment(rgb, model["id"], "whole")
            pred = cv2.resize(pred, (inference.PATCH_SIZE, inference.PATCH_SIZE), interpolation=cv2.INTER_NEAREST)
            truth = load_ground_truth(os.path.join(mask_dir, f"mask_{i}.tiff"))
            confusion += np.bincount(n * truth.ravel() + pred.ravel(), minlength=n * n).reshape(n, n)
        tp = np.diag(confusion)
        dice = 2 * tp / np.maximum(confusion.sum(0) + confusion.sum(1), 1)
        print(f"{model['id']:16s}" + "".join(f"{d:14.3f}" for d in dice) + f"{dice[1:].mean():12.3f}")


if __name__ == "__main__":
    main()
