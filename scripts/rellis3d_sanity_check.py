#!/usr/bin/env python3
"""
Dataset Integrity Verification for RELLIS-3D

This script performs sanity checks on RELLIS-3D dataset structure and content,
verifying that all required files and directories exist across sequences.

Main checks:
1. Core dataset structure (images, annotations, LiDAR, calibration)
   - RGB images (pylon_camera_node/*.jpg)
   - Semantic ID masks (pylon_camera_node_label_id/*.png)
   - Color semantic masks (pylon_camera_node_label_color/*.png)
   - LiDAR point clouds (os1_cloud_node_kitti_bin/*.bin)
   - Sensor calibration (poses.txt, camera_info.txt)

2. Transformation matrices for multi-sensor fusion
   - Basler-to-Ouster extrinsic transforms (transforms.yaml)

3. File consistency
   - Matching stems between images and masks
   - Non-empty pose and camera info files
   - Valid YAML structure in transform files

Usage:
  python rellis3d_sanity_check.py [RELLIS_ROOT]

Arguments:
  RELLIS_ROOT  Path to RELLIS-3D dataset root (default: ~/research-project/RELLIS-3D)

Output: Prints per-sequence summary with OK/MISSING/BAD indicators
Exit code: 0 if all sequences valid, 1 if issues found
"""

import sys
import yaml
from pathlib import Path

# Default dataset root path (can be overridden by command-line argument)
ROOT = (Path(sys.argv[1]) if len(sys.argv) > 1 else Path.home() / "research-project/RELLIS-3D")
rellis_3d = ROOT / "Rellis-3D"
rellis_3D = ROOT / "Rellis_3D"

def stems(p: Path, exts):
    """
    Extract stem names of files matching given extensions in a directory.
    
    Args:
        p (Path): Directory to scan
        exts (list): List of file extensions to match (e.g., ['.jpg', '.png'])
    
    Returns:
        set: Set of file stems (names without extension)
    """
    exts = tuple(exts)
    return {f.stem for f in p.glob("*") if f.suffix.lower() in exts}

bad = 0

# ============================================================================
# Check Standard Dataset Structure (Rellis-3D directory)
# ============================================================================
# Verify images, annotations, and sensor data for each sequence

seq_dirs = sorted([p for p in rellis_3d.glob("[0-9]" * 5) if p.is_dir()])
if not seq_dirs:
    print(f"[!] No sequences found at {rellis_3d}")
    sys.exit(2)

print(f"Checking {len(seq_dirs)} sequences under {rellis_3d} …\n")
for seq_dir in seq_dirs:
    seq = seq_dir.name
    # Specify expected directory and file locations
    img_dir = seq_dir / "pylon_camera_node"
    id_dir = seq_dir / "pylon_camera_node_label_id"
    color_dir = seq_dir / "pylon_camera_node_label_color"
    lidar_dir = seq_dir / "os1_cloud_node_kitti_bin"
    poses = seq_dir / "poses.txt"
    caminfo = seq_dir / "camera_info.txt"

    # Check for missing required directories/files
    missing = [
        p.name
        for p in [img_dir, id_dir, color_dir, lidar_dir, poses, caminfo]
        if not p.exists()
    ]
    if missing:
        print(f"[{seq}] MISSING -> {', '.join(missing)}")
        bad += 1
        continue

    # Collect files in each directory
    jpgs = stems(img_dir, [".jpg", ".jpeg"])
    ids = stems(id_dir, [".png"])
    cols = stems(color_dir, [".png"])
    bins = list(lidar_dir.glob("*.bin"))

    # Check for data presence
    mismatches = []
    if not jpgs:
        mismatches.append("no camera JPGs")
    if not ids:
        mismatches.append("no ID PNGs")
    if not cols:
        mismatches.append("no COLOR PNGs")
    if len(bins) == 0:
        mismatches.append("no LiDAR .bin files")

    # Find files that exist in one modality but not the other
    miss_id = sorted(list(jpgs - ids))[:5]
    miss_col = sorted(list(jpgs - cols))[:5]
    extra_id = sorted(list(ids - jpgs))[:5]
    extra_col = sorted(list(cols - jpgs))[:5]

    # Validate poses and camera info files
    pose_ok = poses.stat().st_size > 0 and any(
        ch.isdigit() for ch in poses.read_text(errors="ignore")[:200]
    )
    if not pose_ok:
        mismatches.append("poses.txt looks empty/non-numeric")

    cam_ok = caminfo.stat().st_size > 0
    if not cam_ok:
        mismatches.append("camera_info.txt empty")

    # Print summary for this sequence
    print(
        f"[{seq}] imgs={len(jpgs):5d}  idPNG={len(ids):5d}  clrPNG={len(cols):5d}  lidarBIN={len(bins):4d}  "
        f"poses={'OK' if pose_ok else 'BAD'}  caminfo={'OK' if cam_ok else 'BAD'}"
    )
    if miss_id:
        print(f"   - labels missing for (first 5): {', '.join(miss_id)}")
    if miss_col:
        print(f"   - color labels missing for (first 5): {', '.join(miss_col)}")
    if extra_id:
        print(f"   - extra ID PNGs (no JPG): {', '.join(extra_id)}")
    if extra_col:
        print(f"   - extra COLOR PNGs (no JPG): {', '.join(extra_col)}")
    if mismatches:
        print(f"   - issues: {', '.join(mismatches)}")
        bad += 1

# ============================================================================
# Check Transformation Data Structure (Rellis_3D directory)
# ============================================================================
# Verify extrinsic calibration files for sensor fusion

seq_dirs_b = sorted([p for p in rellis_3D.glob("[0-9]" * 5) if p.is_dir()])
print(f"Checking {len(seq_dirs_b)} sequences under {rellis_3D} …\n")
for seq_dir in seq_dirs_b:
    seq = seq_dir.name
    tf_yaml = seq_dir / "transforms.yaml"
    mismatches = []
    tf_ok = False
    try:
        y = yaml.safe_load(tf_yaml.read_text())
        # Accept either direct q/t dict or nested under a single key
        if isinstance(y, dict) and y:
            # case 1: top-level has q and t
            if (
                "q" in y
                and "t" in y
                and isinstance(y["q"], dict)
                and isinstance(y["t"], dict)
            ):
                tf_ok = True
            else:
                # case 2: first mapping contains q/t
                first = next(iter(y.values()))
                if isinstance(first, dict) and "q" in first and "t" in first:
                    tf_ok = True
    except Exception:
        tf_ok = False

    print(f"[{seq}] tf.yaml={'OK' if tf_ok else 'BAD'}")
    if not tf_ok:
        print("   - issues: transforms.yaml not parseable/empty")
        bad += 1

print("\nDone.", "ALL GOOD ✅" if bad == 0 else f"{bad} sequence(s) need fixes ❗")
sys.exit(0 if bad == 0 else 1)
