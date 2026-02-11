#!/usr/bin/env python3
"""
RELLIS-3D -> Grounded-SAM (single process, model loaded once).

- Imports GroundingDINO & SAM directly (no subprocess).
- Loads both models on CUDA one time.
- Prepares a union caption from a comma-separated prompt list.
- Iterates all frames under Rellis-3D/*/pylon_camera_node/*.jpg
- Pairs each with pylon_camera_node_label_id/<same>.png
- Produces pred_traversable.png (0/255) per frame, resumably.
- Maintains/updates a dataset-wide index.csv at --out-root.

mAP note: the demo stack doesn’t emit probabilities; we save a
binary union mask. (mAP is therefore not meaningful here.)
"""

from __future__ import annotations
import argparse
import csv
import os
import sys
from pathlib import Path
from typing import Iterable, Optional, Tuple


# ---------- headless hygiene (silence Qt/WSLg) ----------
def ensure_offscreen_env() -> None:
    os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
    xdg = os.environ.get(
        "XDG_RUNTIME_DIR", f"/tmp/runtime-{os.environ.get('USER','user')}"
    )
    os.environ["XDG_RUNTIME_DIR"] = xdg
    Path(xdg).mkdir(parents=True, exist_ok=True)
    try:
        os.chmod(xdg, 0o700)
    except PermissionError:
        pass


ensure_offscreen_env()

# ---------- add repo root to sys.path so imports work ----------
# pre-parse just --gsam-root so we can set sys.path before heavy imports
_pre = argparse.ArgumentParser(add_help=False)
_pre.add_argument("--gsam-root", default=os.environ.get("GSAM_ROOT", ""))
args_known, _ = _pre.parse_known_args()

if args_known.gsam_root:
    gs = Path(args_known.gsam_root).resolve()
    sys.path.insert(0, str(gs))
else:
    # fallback: use current working dir (useful if you cd into the repo)
    sys.path.insert(0, str(Path.cwd()))

# ---------- deps ----------
import numpy as np
import cv2
import torch

# GroundingDINO convenience API
from GroundingDINO.groundingdino.util.inference import (
    load_model as gd_load_model,
    load_image as gd_load_image,
    predict as gd_predict,
)

# SAM predictor
from segment_anything.predictor import SamPredictor
from segment_anything.build_sam import sam_model_registry


# ---------------------------- pairing ----------------------------


def exact_id_pair_for_image(img_path: Path) -> Path:
    """Exact mapping for your layout:
    .../<seq>/pylon_camera_node/<stem>.jpg
    -> .../<seq>/pylon_camera_node_label_id/<stem>.png
    """
    seq_root = img_path.parents[1]
    stem = img_path.stem
    return seq_root / "pylon_camera_node_label_id" / f"{stem}.png"


def fuzzy_find_id_pair(img_path: Path) -> Optional[Path]:
    """
    Fallback: search within the sequence directory for <stem>.png and
    score candidates that live under *label* and *id* subdirs higher.
    Used only if the exact path is missing.
    """
    seq_root = img_path.parents[1]
    stem = img_path.stem
    best: Tuple[int, Optional[Path]] = (-1, None)
    for p in seq_root.rglob(f"{stem}.png"):
        s = 0
        low = str(p.parent).lower()
        if "label" in low:
            s += 2
        if "id" in low:
            s += 3
        if "color" in low:
            s -= 3
        if "pylon" in low:
            s += 1
        if s > best[0]:
            best = (s, p)
    return best[1]


def find_gt_id_mask(img_path: Path) -> Optional[Path]:
    """Try exact mapping first; if absent, fall back to fuzzy search."""
    id_path = exact_id_pair_for_image(img_path)
    if id_path.exists():
        return id_path
    return fuzzy_find_id_pair(img_path)


# ----------------------- model construction ----------------------


def build_groundingdino(cfg_path: Path, ckpt_path: Path, device: str):
    """
    Load and initialize GroundingDINO model.
    
    GroundingDINO is used for object detection of traversable regions based on
    text prompts (e.g., "road, grass, gravel, path").

    Args:
        cfg_path (Path): Path to GroundingDINO config YAML file
        ckpt_path (Path): Path to GroundingDINO checkpoint weights
        device (str): Device to load model on (e.g., "cuda", "cpu")

    Returns:
        GroundingDINO model in eval mode on specified device
    """
    model = gd_load_model(str(cfg_path), str(ckpt_path), device=device)
    model.eval()
    return model


def build_sam(sam_ckpt: Path, device: str, sam_type: str = "vit_h") -> SamPredictor:
    """
    Load and initialize Segment Anything Model (SAM) predictor.
    
    SAM performs instance-level segmentation of regions proposed by GroundingDINO.

    Args:
        sam_ckpt (Path): Path to SAM checkpoint weights
        device (str): Device to load model on (e.g., "cuda", "cpu")
        sam_type (str): SAM model variant. Common options: "vit_h", "vit_l", "vit_b".
                       Must match the checkpoint architecture.

    Returns:
        SamPredictor: Initialized SAM predictor wrapper for mask generation
    """
    # sam_type must match your checkpoint; default fits sam_vit_h_4b8939.pth
    sam = sam_model_registry[sam_type](checkpoint=str(sam_ckpt))
    sam.to(device=device)
    sam.eval()
    return SamPredictor(sam)


# ====== Data Iteration & Utility Functions ======


def iter_rellis_images(rellis_root: Path) -> Iterable[Path]:
    """
    Iterate over all RGB camera images from RELLIS-3D sequences.
    
    Directory structure: RELLIS-3D/<seq>/pylon_camera_node/*.jpg
    Files are returned in sorted order for reproducibility.

    Args:
        rellis_root (Path): Path to RELLIS-3D root directory

    Yields:
        Path: Sorted paths to RGB image files (.jpg)
    """
    # Your tree: Rellis-3D/<seq>/pylon_camera_node/*.jpg
    return sorted(rellis_root.glob("*/pylon_camera_node/*.jpg"))


# ====== Mask Utilities ======


def binarize_to_png(mask_union: np.ndarray, out_png: Path) -> Path:
    """
    Save a boolean mask array as a 0/255 PNG image.
    
    Converts boolean or 0-1 floating point masks to standard 8-bit PNG format.
    Creates parent directories if they don't exist.

    Args:
        mask_union (np.ndarray): Input mask as (H, W) boolean/0-1 array
        out_png (Path): Output PNG file path

    Returns:
        Path: Path to written PNG file
    """
    if mask_union.dtype != np.bool_:
        mask_union = mask_union.astype(bool)
    out_png.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(out_png), (mask_union.astype(np.uint8) * 255))
    return out_png


def _sigmoid(x: float) -> float:
    return 1.0 / (1.0 + np.exp(-float(x)))


def draw_labeled_box(img: np.ndarray, box_xyxy, label: str, color=(0, 255, 0)) -> None:
    """
    Draw a labeled bounding box on an RGB image in-place.
    
    Creates a rectangle with a high-contrast filled label box containing text.
    Useful for visualizing GroundingDINO detection boxes with confidence scores.

    Args:
        img (np.ndarray): RGB image array (H, W, 3) to draw on (modified in-place)
        box_xyxy: Bounding box coordinates [x1, y1, x2, y2] (pixel units)
        label (str): Text label to display (e.g., "road 0.95")
        color (tuple): RGB box color (default: green (0, 255, 0))

    Returns:
        None (image is modified in-place)
    """
    x1, y1, x2, y2 = [int(v) for v in box_xyxy]
    cv2.rectangle(img, (x1, y1), (x2, y2), color, 2)

    # Text box
    font = cv2.FONT_HERSHEY_SIMPLEX
    font_scale = 0.7
    thickness = 2
    (tw, th), baseline = cv2.getTextSize(label, font, font_scale, thickness)
    pad = 3
    # Filled background rectangle for the text
    cv2.rectangle(
        img, (x1, max(0, y1 - th - 2 * pad)), (x1 + tw + 2 * pad, y1), color, -1
    )
    # Black text on colored background
    cv2.putText(
        img,
        label,
        (x1 + pad, y1 - pad),
        font,
        font_scale,
        (0, 0, 0),
        thickness,
        cv2.LINE_AA,
    )


# -------------------- box conversion -------------------


def cxcywh_to_xyxy(boxes_xywh: torch.Tensor) -> torch.Tensor:
    """
    Convert bounding box format from center coordinates to corner coordinates.
    
    Converts from [cx, cy, width, height] normalized format (GroundingDINO output)
    to [x1, y1, x2, y2] corner format (required by SAM predictor).

    Args:
        boxes_xywh (torch.Tensor): Box tensor of shape (N, 4) in [cx, cy, w, h] format

    Returns:
        torch.Tensor: Box tensor of shape (N, 4) in [x1, y1, x2, y2] format
    """
    cx = boxes_xywh[:, 0]
    cy = boxes_xywh[:, 1]
    w = boxes_xywh[:, 2]
    h = boxes_xywh[:, 3]
    x1 = cx - 0.5 * w
    y1 = cy - 0.5 * h
    x2 = cx + 0.5 * w
    y2 = cy + 0.5 * h
    return torch.stack([x1, y1, x2, y2], dim=1)


# ==============================================================================
# Main Processing Pipeline
# ==============================================================================


def main():
    """
    Main entry point for single-process RELLIS-3D prediction pipeline.
    
    High-level workflow:
    1. Parse arguments and initialize models (GroundingDINO + SAM)
    2. Prepare output directory and index CSV for resumable processing
    3. For each image:
       a. Run GroundingDINO text detection to get traversable region boxes
       b. Run SAM on those boxes to get precise instance masks
       c. Union all masks and save as pred_traversable.png
       d. Update index.csv with path and metadata
    4. Write final index.csv with all predictions
    
    Resumable execution: Skips frames where pred_traversable.png already exists
    unless --force flag is provided.
    """
    ap = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    ap.add_argument(
        "--rellis-root",
        type=Path,
        required=True,
        help="Path to .../RELLIS-3D/Rellis-3D",
    )
    ap.add_argument(
        "--gsam-root",
        type=Path,
        required=True,
        help="Path to your Grounded-Segment-Anything repo root",
    )
    ap.add_argument("--dino-config", type=Path, required=True)
    ap.add_argument("--dino-ckpt", type=Path, required=True)
    ap.add_argument("--sam-ckpt", type=Path, required=True)
    ap.add_argument(
        "--sam-type",
        type=str,
        default="vit_h",
        help="SAM model type key for sam_model_registry",
    )
    ap.add_argument(
        "--prompts",
        type=str,
        default="road, dirt, gravel, grass, trail, path, mud, ground",
        help="Comma-separated list treated as a union prompt",
    )
    ap.add_argument("--box-threshold", type=float, default=0.30)
    ap.add_argument("--text-threshold", type=float, default=0.25)
    ap.add_argument("--device", type=str, default="cuda")
    ap.add_argument(
        "--out-root",
        type=Path,
        required=True,
        help="Output root. We write <seq>/<stem>/pred_traversable.png",
    )
    ap.add_argument(
        "--limit",
        type=int,
        default=0,
        help="Process at most N *new* predictions (respects saved empties too).",
    )
    ap.add_argument(
        "--overlay",
        action="store_true",
        help="Save an overlay with predicted boxes per frame",
    )
    ap.add_argument(
        "--force",
        action="store_true",
        help="Recompute even if pred_traversable.png already exists",
    )
    args = ap.parse_args()

    # GroundingDINO likes sentence-separated phrases (periods).
    parts = [t.strip() for t in args.prompts.split(",") if t.strip()]
    caption = ". ".join(parts)
    if not caption.endswith("."):
        caption += "."

    # Initialize models once (loaded on GPU, reused for all frames)
    print("[*] Loading GroundingDINO…")
    gd_model = build_groundingdino(args.dino_config, args.dino_ckpt, args.device)
    print("[*] Loading SAM…")
    sam_predictor = build_sam(args.sam_ckpt, args.device, args.sam_type)

    # Load or create index for resumable processing
    index_csv = args.out_root / "index.csv"
    index_rows = {}  # key -> dict
    if index_csv.exists():
        with index_csv.open() as f:
            reader = csv.DictReader(f)
            for r in reader:
                index_rows[r["pred_path"]] = r

    processed = 0  # counts new predictions we saved this run
    for img_path in iter_rellis_images(args.rellis_root):
        seq = img_path.parents[1].name
        stem = img_path.stem
        work_dir = args.out_root / seq / stem
        pred_png = work_dir / "pred_traversable.png"

        # Resume: skip if prediction already exists (unless --force)
        if pred_png.exists() and not args.force:
            key = str(pred_png.relative_to(args.out_root))
            if key not in index_rows:
                gt_id = find_gt_id_mask(img_path)
                index_rows[key] = {
                    "sequence": seq,
                    "frame": stem,
                    "image_path": str(img_path),
                    "gt_id_path": str(gt_id) if gt_id else "",
                    "pred_path": key,
                    "status": "ok",
                }
            continue

        # Pair GT id mask (for later evaluation)
        gt_id = find_gt_id_mask(img_path)
        if gt_id is None:
            print(f"[warn] Missing GT id for {img_path}; continuing.")

        # ========== GroundingDINO Text Detection ==========
        # Load image in RGB format and network tensor format
        image_source, image_net = gd_load_image(
            str(img_path)
        )  # RGB uint8 + network tensor
        H, W = image_source.shape[:2]

        # Detect traversable regions using text prompts
        with torch.inference_mode():
            boxes, logits, phrases = gd_predict(
                model=gd_model,
                image=image_net,
                caption=caption,
                box_threshold=args.box_threshold,
                text_threshold=args.text_threshold,
                device=args.device,
            )
            num_boxes = (
                0
                if boxes is None
                else (boxes.shape[0] if hasattr(boxes, "shape") else len(boxes))
            )
            print(f"[info] {seq}/{stem}: boxes={num_boxes}, HxW={H}x{W}")

        # Handle case: no boxes detected (empty prediction)
        if boxes is None or (hasattr(boxes, "shape") and boxes.shape[0] == 0):
            binarize_to_png(np.zeros((H, W), dtype=bool), pred_png)
            key = str(pred_png.relative_to(args.out_root))
            index_rows[key] = {
                "sequence": seq,
                "frame": stem,
                "image_path": str(img_path),
                "gt_id_path": str(gt_id) if gt_id else "",
                "pred_path": key,
                "status": "no_boxes",
            }
            processed += 1
            if args.limit and processed >= args.limit:
                break
            continue

        # ========== Convert & Validate Boxes ==========
        # Convert GroundingDINO output format to SAM input format
        if isinstance(boxes, np.ndarray):
            boxes = torch.from_numpy(boxes)
        boxes = boxes.float().to(args.device)  # normalized [cx,cy,w,h] in [0,1]

        # scale to pixel coordinates
        scale = torch.tensor([W, H, W, H], device=boxes.device, dtype=boxes.dtype)
        boxes = boxes * scale
        # Convert from center coords [cx,cy,w,h] to corner coords [x1,y1,x2,y2]
        boxes = cxcywh_to_xyxy(boxes)

        # clamp to image bounds
        boxes[:, 0::2].clamp_(0, W - 1)  # x1,x2
        boxes[:, 1::2].clamp_(0, H - 1)  # y1,y2

        # Drop extremely small boxes (rare but improves quality)
        wh = (boxes[:, 2] - boxes[:, 0]).clamp_min(0) * (
            boxes[:, 3] - boxes[:, 1]
        ).clamp_min(0)
        keep = wh > 4.0
        if keep.sum() == 0:
            # No valid boxes after filtering
            binarize_to_png(np.zeros((H, W), dtype=bool), pred_png)
            key = str(pred_png.relative_to(args.out_root))
            index_rows[key] = {
                "sequence": seq,
                "frame": stem,
                "image_path": str(img_path),
                "gt_id_path": str(gt_id) if gt_id else "",
                "pred_path": key,
                "status": "no_valid_boxes",
            }
            processed += 1
            if args.limit and processed >= args.limit:
                break
            continue

        # Subset boxes and keep corresponding metadata
        idx_keep = keep.nonzero(as_tuple=False).squeeze(1).tolist()
        boxes = boxes[keep]
        phr_kept = (
            [phrases[i] for i in idx_keep]
            if phrases is not None
            else ["object"] * boxes.shape[0]
        )
        scores_kept = []
        if logits is not None:
            # logits can be tensor/ndarray/list; normalize and subset
            if hasattr(logits, "detach"):
                logits_np = logits.detach().cpu().numpy()
            else:
                logits_np = np.array(logits)
            scores_kept = [_sigmoid(logits_np[i]) for i in idx_keep]
        else:
            scores_kept = [1.0] * boxes.shape[0]

        # ========== SAM Instance Segmentation ==========
        # Run SAM on all detected boxes to get precise masks

        if args.overlay:
            overlay = image_source.copy()
            b_np = boxes.detach().to("cpu").numpy()
            for i, b in enumerate(b_np):
                label_text = f"{phr_kept[i]} {scores_kept[i]:.2f}"
                draw_labeled_box(overlay, b, label_text, color=(0, 255, 0))
            work_dir.mkdir(parents=True, exist_ok=True)
            cv2.imwrite(
                str(work_dir / "boxes_overlay.jpg"),
                cv2.cvtColor(overlay, cv2.COLOR_RGB2BGR),
            )

        # map to SAM's internal space, predict masks
        # Transform box coordinates to SAM's internal coordinate space
        transformed = sam_predictor.transform.apply_boxes_torch(boxes, (H, W))
        # Run SAM to get masks for each box
        masks, _, _ = sam_predictor.predict_torch(
            point_coords=None,
            point_labels=None,
            boxes=transformed,
            multimask_output=False,
        )  # (N,1,H,W)

        # Union all mask predictions (traversable = any box predicts traversable)
        union = masks.squeeze(1).any(dim=0).cpu().numpy().astype(bool)

        # ========== Post-processing ==========
        # Apply morphological operations to smooth predictions
        kernel = np.ones((5, 5), np.uint8)
        union = cv2.morphologyEx(
            union.astype(np.uint8) * 255, cv2.MORPH_CLOSE, kernel, iterations=1
        )
        union = cv2.morphologyEx(union, cv2.MORPH_OPEN, kernel, iterations=1)
        union = union > 0

        # Save union as 0/255 PNG
        binarize_to_png(union, pred_png)

        # Update index with this prediction (enables resumable processing)
        key = str(pred_png.relative_to(args.out_root))
        index_rows[key] = {
            "sequence": seq,
            "frame": stem,
            "image_path": str(img_path),
            "gt_id_path": str(gt_id) if gt_id else "",
            "pred_path": key,
            "status": "ok",
        }
        processed += 1
        if args.limit and processed >= args.limit:
            break

    # Write/refresh index.csv atomically
    index_csv.parent.mkdir(parents=True, exist_ok=True)
    with index_csv.open("w", newline="") as f:
        cols = ["sequence", "frame", "image_path", "gt_id_path", "pred_path", "status"]
        w = csv.DictWriter(f, fieldnames=cols)
        w.writeheader()
        for _, row in sorted(index_rows.items()):
            w.writerow(row)

    print(f"[*] Done. Predictions: {args.out_root}")
    print(f"[*] Index: {index_csv}")


if __name__ == "__main__":
    main()
