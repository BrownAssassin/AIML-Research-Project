#!/usr/bin/env bash
set -euo pipefail

# ==============================================================================
# Download RELLIS-3D Dataset Components
#
# This script downloads all RELLIS-3D dataset components from Google Drive,
# including RGB images, semantic ID annotations, color annotations, image splits,
# LiDAR point clouds (Ouster format), camera poses, calibration parameters, and
# sensor-to-sensor transformation matrices.
#
# Usage:
#   ./download_rellis3d_from_readme.sh [OUTPUT_DIR]
#
# Arguments:
#   OUTPUT_DIR  Destination directory (default: $HOME/research-project/RELLIS-3D)
#
# Prerequisites:
#   - python, unzip, tar, rsync must be available on PATH
#   - gdown will be auto-installed via pip
#
# Output Structure:
#   Created under OUTPUT_DIR with extracted subdirectories for:
#   - RELLIS-3D/ (images + annotations)
#   - ouster_lidar_semantickitti/ (LiDAR point clouds in KITTI format)
#   - ouster_lidar_scan_poses/ (Sensor poses for each scan)
#   - camera_intrinsic/ (Camera calibration parameters)
#   - basler_to_ouster/ (Transformation matrices)
# ==============================================================================

OUT_DIR="${1:-$HOME/research-project/RELLIS-3D}"
mkdir -p "$OUT_DIR"

# Check for required system dependencies (fail gracefully if missing)
need() { command -v "$1" >/dev/null 2>&1 || { echo "[-] Missing: $1"; exit 1; }; }
need python; need unzip; need tar; need rsync
echo "[*] Installing gdown…"
python -m pip install -q --upgrade gdown >/dev/null

cd "$OUT_DIR"

# ============================================================================
# Google Drive URLs for dataset components
# Note: These are the actual download links; view= URLs are fuzzy-matched by gdown.
# ============================================================================
# Standard dataset components
IMAGES_URL='https://drive.google.com/file/d/1F3Leu0H_m6aPVpZITragfreO_SGtL2yV/view'       # Full RGB images (~11GB)
IDS_URL='https://drive.google.com/file/d/16URBUQn_VOGvUqfms-0I8HHKMtjPHsu5/view'         # Semantic ID masks
COLOR_URL='https://drive.google.com/file/d/1HJl8Fi5nAjOr41DPUFmkeKWtDXhCZDke/view'       # Color-coded semantic masks
SPLIT_URL='https://drive.google.com/file/d/1zHmnVaItcYJAWat3Yti1W_5Nfux194WQ/view'       # Train/val/test split definitions

IMAGES_ZIP="$OUT_DIR/RELLIS-3D-images.zip"
IDS_ZIP="$OUT_DIR/RELLIS-3D-annotations-id.zip"
COLOR_ZIP="$OUT_DIR/RELLIS-3D-annotations-color.zip"
SPLIT_ZIP="$OUT_DIR/RELLIS-3D-splits.zip"

# LiDAR and calibration components (for preprocessing with Follow-the-Footprints)
# Note: Replace these placeholder URLs with actual Google Drive share links from README

LIDAR_SEMKITTI_URL='https://drive.google.com/file/d/1lDSVRf_kZrD0zHHMsKJ0V1GN9QATR4wH/view'
POSES_URL='https://drive.google.com/file/d/1V3PT_NJhA41N7TBLp5AbW31d0ztQDQOX/view'
CAM_INTRINSIC_URL='https://drive.google.com/file/d/1NAigZTJYocRSOTfgFBddZYnDsI_CSpwK/view'
BASLER2OUSTER_URL='https://drive.google.com/file/d/19EOqWS9fDUFp4nsBrMCa69xs9LgIlS2e/view'

LIDAR_ARC="$OUT_DIR/ouster_lidar_semantickitti.zip"
POSES_ARC="$OUT_DIR/ouster_lidar_scan_poses.zip"
CAM_INTR_ARC="$OUT_DIR/camera_intrinsic.zip"          # May contain .txt, .yaml, or both
B2O_ARC="$OUT_DIR/basler_to_ouster.zip"               # Basler-to-Ouster extrinsic calibration

# ============================================================================
# Helper Functions
# ============================================================================

# Download a file from Google Drive if not already present.
# Args:
#   $1 - Google Drive URL (in 'view' format, fuzzy-matched by gdown)
#   $2 - Destination file path
download() {
  local url="$1" dest="$2"
  if [[ -z "${url// }" ]]; then
    # Skip empty URLs (accommodate multiple data sources)
    echo "[*] Skipping empty URL for $(basename "$dest")"
    return 0
  fi
  if [[ -e "$dest" ]]; then
    # Resume: only skip download if archive already exists
    echo "[*] Found $(basename "$dest") — skipping download."
  else
    echo "[*] Downloading $(basename "$dest")"
    gdown --fuzzy -c "$url" -O "$dest"
  fi
}

# Extract an archive file (zip, tar.xz, tar.gz, tar) if not already extracted.
# Uses a stamp file to track extraction state for idempotency.
# Args:
#   $1 - Archive file path (.zip, .tar.gz, .tar.xz, or .tar)
extract_once() {
  local a="$1" base stamp
  [[ -e "$a" ]] || return 0
  base=$(basename "$a")
  stamp="$OUT_DIR/.extracted-${base}.ok"
  if [[ -e "$stamp" ]]; then
    # Already extracted (marked by stamp file)
    echo "[*] Already extracted: $base"
    return 0
  fi
  echo "[*] Extracting: $base"
  case "$a" in
    *.zip)      unzip -qq -n "$a" -d "$OUT_DIR" ;;
    *.tar.xz)   tar -xJf "$a" -C "$OUT_DIR" ;;
    *.tar.gz|*.tgz) tar -xzf "$a" -C "$OUT_DIR" ;;
    *.tar)      tar -xf "$a" -C "$OUT_DIR" ;;
    *)          echo "[!] Unknown archive type: $a"; return 1 ;;
  esac
  touch "$stamp"
}

# Find and copy the first file matching a pattern from source to destination.
# Useful for extracting single files from collections.
# Args:
#   $1 - Source directory
#   $2 - Case-insensitive filename pattern (e.g., "*.txt")
#   $3 - Destination file path
copy_first_file_like() {
  local src_dir="$1" pattern="$2" dest="$3"
  local f
  f=$(find "$src_dir" -type f -iname "$pattern" | head -n1 || true)
  [[ -n "$f" ]] && cp "$f" "$dest"
}

# ============================================================================
# Main Download and Extraction Pipeline (Idempotent)
# ============================================================================
download "$IMAGES_URL" "$IMAGES_ZIP"
download "$IDS_URL"    "$IDS_ZIP"
download "$COLOR_URL"  "$COLOR_ZIP"
download "$SPLIT_URL"  "$SPLIT_ZIP"

download "$LIDAR_SEMKITTI_URL" "$LIDAR_ARC"
download "$POSES_URL"          "$POSES_ARC"
download "$CAM_INTRINSIC_URL"  "$CAM_INTR_ARC"
download "$BASLER2OUSTER_URL"  "$B2O_ARC"

# ---------------- extracts ----------------
extract_once "$IMAGES_ZIP"
extract_once "$IDS_ZIP"
extract_once "$COLOR_ZIP"
extract_once "$SPLIT_ZIP"

extract_once "$LIDAR_ARC"
extract_once "$POSES_ARC"

# Note: Camera intrinsics and extrinsic transforms may not be archives; extraction will skip if file is not a zip.

# ============================================================================
# Organize Files for Follow-the-Footprints (FTF) Preprocessing
# This section arranges downloaded files into the expected directory structure
# for downstream preprocessing pipelines.
# ============================================================================

mkdir -p "$OUT_DIR/Rellis-3D" "$OUT_DIR/Rellis_3D"

# Copy LiDAR point cloud .bin files to each sequence directory
# Creates: Rellis-3D/<seq>/os1_cloud_node_kitti_bin/
echo "[*] Placing LiDAR .bin files…"
if [[ -e "$LIDAR_ARC" ]]; then
  while IFS= read -r -d '' seqdir; do
    seq=$(basename "$seqdir")
    dst="$OUT_DIR/Rellis-3D/$seq/os1_cloud_node_kitti_bin"
    mkdir -p "$dst"
    # Copy all .bin files; works regardless of source folder naming (velodyne, lidar, os1_*, etc.)
    find "$seqdir" -type f -name '*.bin' -print0 \
      | rsync -a --files-from=- --from0 / "$dst"/
  done < <(find "$OUT_DIR" -type d -regex '.*/[0-9]{5}$' -print0)
fi

# Copy pose files to each sequence (convert CSV to space-delimited if needed)
# Creates: Rellis-3D/<seq>/poses.txt
echo "[*] Placing pose files…"

# 2) Pose files → Rellis-3D/<seq>/poses.txt  (CSV → space-delimited)
if [[ -e "$POSES_ARC" ]]; then
  echo "[*] Placing pose files…"
  while IFS= read -r -d '' seqdir; do
    seq=$(basename "$seqdir")
    # Locate pose file matching this sequence number
    pose_cand=$(find "$OUT_DIR" -maxdepth 4 -type f \( -iname "*${seq}*.txt" -o -iname "*${seq}*.csv" -o -iname "*${seq}*pose*" \) | head -n1 || true)
    if [[ -n "$pose_cand" ]]; then
      # Convert CSV to space-delimited format if needed
      if [[ "$pose_cand" =~ \.csv$ ]]; then
        sed 's/,/ /g' "$pose_cand" > "$OUT_DIR/Rellis-3D/$seq/poses.txt"
      else
        cp "$pose_cand" "$OUT_DIR/Rellis-3D/$seq/poses.txt"
      fi
    else
      echo "[warn] No pose file found for sequence $seq"
    fi
  done < <(find "$OUT_DIR/Rellis-3D" -mindepth 1 -maxdepth 1 -type d -regex '.*/[0-9]{5}$' -print0)
fi

# Copy camera intrinsic calibration parameters to each sequence
# Creates: Rellis-3D/<seq>/camera_info.txt
echo "[*] Copying camera intrinsics to each sequence…"
if [[ -s "$CAM_INTR_ARC" ]]; then
  tmp_intr="$OUT_DIR/.tmp_intr"
  mkdir -p "$tmp_intr"
  unzip -qq -n "$CAM_INTR_ARC" -d "$tmp_intr" || true
  # Find intrinsic file (extracted or direct upload)
  intr_src=$( (ls "$CAM_INTR_ARC" 2>/dev/null || true) && echo "$CAM_INTR_ARC" )   # default
  cand=$(find "$tmp_intr" -type f | head -n1 || true)
  [[ -n "$cand" ]] && intr_src="$cand"
  while IFS= read -r -d '' seqdir; do
    cp "$intr_src" "$seqdir/camera_info.txt" || true
  done < <(find "$OUT_DIR/Rellis-3D" -mindepth 1 -maxdepth 1 -type d -regex '.*/[0-9]{5}$' -print0)
fi

# Copy Basler-to-Ouster sensor calibration (extrinsic transforms) to each sequence
# Creates: Rellis_3D/<seq>/transforms.yaml
echo "[*] Copying transforms.yaml to Rellis_3D/<seq>/…"
if [[ -s "$B2O_ARC" ]]; then
  tmp_b2o="$OUT_DIR/.tmp_b2o"
  mkdir -p "$tmp_b2o"
  unzip -qq -n "$B2O_ARC" -d "$tmp_b2o" || true
  # Find transforms.yaml (extracted or direct upload)
  yaml_src=$(find "$tmp_b2o" -type f -iname 'transforms.yaml' -o -iname '*.yaml' | head -n1 || true)
  [[ -z "$yaml_src" && -f "$B2O_ARC" && "${B2O_ARC##*.}" = "yaml" ]] && yaml_src="$B2O_ARC"
  if [[ -n "$yaml_src" ]]; then
    while IFS= read -r -d '' seqdir; do
      seq=$(basename "$seqdir")
      mkdir -p "$OUT_DIR/Rellis_3D/$seq"
      cp "$yaml_src" "$OUT_DIR/Rellis_3D/$seq/transforms.yaml"
    done < <(find "$OUT_DIR/Rellis-3D" -mindepth 1 -maxdepth 1 -type d -regex '.*/[0-9]{5}$' -print0)
  else
    echo "[warn] Could not locate transforms.yaml in $B2O_ARC"
  fi
fi

# ============================================================================
# Sanity Check: Verify Dataset Completeness
# ============================================================================
echo ""
echo "---- Sanity Check (Sample Sequences) ----"
for s in 00000 00001 00002; do
  if [[ -d "$OUT_DIR/Rellis-3D/$s" ]]; then
    echo "$s:"
    ls -1 "$OUT_DIR/Rellis-3D/$s" | sed 's/^/  /'
    test -d "$OUT_DIR/Rellis-3D/$s/os1_cloud_node_kitti_bin" && echo "  LiDAR bin ✅"
    test -f "$OUT_DIR/Rellis-3D/$s/poses.txt" && echo "  poses.txt ✅"
    test -f "$OUT_DIR/Rellis-3D/$s/camera_info.txt" && echo "  camera_info.txt ✅"
    test -f "$OUT_DIR/Rellis_3D/$s/transforms.yaml" && echo "  transforms.yaml ✅" || echo "  transforms.yaml MISSING"
  fi
done

echo "[✓] Download and extraction complete. Data is in: $OUT_DIR"
