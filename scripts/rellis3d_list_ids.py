#!/usr/bin/env python3
"""
Scan RELLIS-3D id/color masks, list all ID values with counts,
compute MEAN COLOR *for that ID only* (masked), and estimate how
"ground-like" an ID is via bottom_fraction: the fraction of that ID's
pixels that lie in the bottom 35% of the image.

Outputs: rellis_ids.csv with:
id,count,mean_r,mean_g,mean_b,bottom_fraction,example_id_png,example_color_png
"""

from pathlib import Path
import argparse, csv
import numpy as np, cv2


def pair_paths(seq_dir: Path):
    """
    Find and yield pairs of ID masks and color masks from a sequence directory.
    
    Matches ID PNG files with corresponding color PNG files by stem (filename).
    Only yields pairs where both files exist.

    Args:
        seq_dir (Path): Sequence directory (contains pylon_camera_node_label_id and
                        pylon_camera_node_label_color subdirectories)

    Yields:
        Tuple[Path, Path]: (id_png_path, color_png_path) pairs
    """


def main():
    """
    Main entry point for semantic ID analysis.
    
    Usage:
      python rellis3d_list_ids.py --rellis-root <path/to/Rellis-3D> [--out-csv output.csv] [--max-seqs N]
    
    Computes and outputs per-ID statistics including color and ground-likeness metrics.
    """
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--rellis-root", required=True, type=Path, help=".../RELLIS-3D/Rellis-3D"
    )
    ap.add_argument("--out-csv", type=Path, default=Path("rellis_ids.csv"))
    ap.add_argument("--max-seqs", type=int, default=0, help="0 = all sequences")
    args = ap.parse_args()

    rows = {}  # id -> dict with counters and running color sums
    seq_dirs = sorted([p for p in args.rellis_root.iterdir() if p.is_dir()])
    if args.max_seqs > 0:
        seq_dirs = seq_dirs[: args.max_seqs]

    for sd in seq_dirs:
        id_dir = sd / "pylon_camera_node_label_id"
        if not id_dir.exists():
            continue
        for id_png, col_png in pair_paths(sd):
            # Load ID mask (uint8 or uint16 semantic labels)
            id_img = cv2.imread(str(id_png), cv2.IMREAD_UNCHANGED)
            if id_img is None:
                continue
            if id_img.ndim == 3:
                id_img = id_img[:, :, 0]
            H, W = id_img.shape[:2]
            # Load corresponding color image
            col_img = cv2.imread(str(col_png), cv2.IMREAD_COLOR)  # BGR
            if col_img is None:
                continue

            # Get all unique IDs and their pixel counts in this frame
            vals, cnts = np.unique(id_img, return_counts=True)
            # Create mask for "ground" hypothesis (bottom 35% of image typically has ground)
            bottom_mask = np.zeros_like(id_img, dtype=bool)
            bottom_mask[int(0.65 * H) :, :] = True

            for v, c in zip(vals.tolist(), cnts.tolist()):
                m = id_img == v
                # Compute mean color for pixels of this ID (masked)
                col = col_img[m]
                if col.size == 0:
                    mb = mg = mr = 0.0
                else:
                    mb, mg, mr = col.mean(axis=0).astype(float)

                # Compute bottom fraction (how many pixels are in the bottom portion)
                bf = (m & bottom_mask).sum() / (m.sum() + 1e-9)

                rec = rows.get(v)
                if rec is None:
                    # First occurrence of this ID
                    rows[v] = dict(
                        id=v,
                        count=c,
                        sum_b=mb * c,
                        sum_g=mg * c,
                        sum_r=mr * c,  # weight by number of pixels
                        sum_bf=bf * c,  # weight by pixels so large regions influence more
                        total=c,
                        example_id_png=str(id_png),
                        example_color_png=str(col_png),
                    )
                else:
                    # Accumulate statistics for existing ID
                    rec["count"] += c
                    rec["sum_b"] += mb * c
                    rec["sum_g"] += mg * c
                    rec["sum_r"] += mr * c
                    rec["sum_bf"] += bf * c
                    rec["total"] += c

    if not rows:
        print("No IDs found. Check --rellis-root.")
        return

    out = args.out_csv
    out.parent.mkdir(parents=True, exist_ok=True)
    with out.open("w", newline="") as f:
        w = csv.DictWriter(
            f,
            fieldnames=[
                "id",
                "count",
                "mean_r",
                "mean_g",
                "mean_b",
                "bottom_fraction",
                "example_id_png",
                "example_color_png",
            ],
        )
        w.writeheader()
        for k in sorted(rows.keys()):
            r = rows[k]
            mean_b = r["sum_b"] / r["total"]
            mean_g = r["sum_g"] / r["total"]
            mean_r = r["sum_r"] / r["total"]
            bottom_fraction = r["sum_bf"] / r["total"]
            w.writerow(
                dict(
                    id=k,
                    count=r["count"],
                    mean_r=mean_r,
                    mean_g=mean_g,
                    mean_b=mean_b,
                    bottom_fraction=bottom_fraction,
                    example_id_png=r["example_id_png"],
                    example_color_png=r["example_color_png"],
                )
            )

    # Console summary focused on "ground-likeness"
    print(f"[*] Wrote {out}")
    top = sorted(rows.values(), key=lambda r: -(r["sum_bf"] / max(r["total"], 1)))[:15]
    print("[*] Likely ground-like IDs (high bottom_fraction):")
    for r in top:
        bf = r["sum_bf"] / max(r["total"], 1)
        print(f"  id={r['id']:>3}  bottom_fraction={bf:.2f}  count={r['count']}")
    print(
        "\nPick the set you consider traversable. You can also eyeball mean colors in the CSV."
    )


if __name__ == "__main__":
    main()
