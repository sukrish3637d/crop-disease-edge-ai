"""
Duplicate detection and dataset cleanup module for Paddy Doctor images.

Provides perceptual-hash-based (pHash) duplicate detection and manifest generation.
"""

from __future__ import annotations

import argparse
import re
from pathlib import Path
from typing import Any

import imagehash
import pandas as pd
from PIL import Image


def _extract_numeric_key(file_path: str) -> tuple[int | float, str]:
    """
    Extract a numeric sort key from a file path stem.

    Returns a tuple of (integer_value, stem_str) to ensure numeric sorting
    (e.g., 99999 comes before 100169), falling back to infinity and the stem
    if no digits are present.
    """
    stem = Path(file_path).stem
    match = re.search(r"\d+", stem)
    if match:
        return (int(match.group()), stem)
    return (float("inf"), stem)


def _to_posix_rel_path(path: Path) -> str:
    """Normalize file path to POSIX relative path from project root."""
    try:
        rel = path.resolve().relative_to(Path.cwd().resolve())
        return rel.as_posix()
    except ValueError:
        if "data" in path.parts:
            idx = path.parts.index("data")
            return "/".join(path.parts[idx:])
        return path.as_posix()


def compute_phash_table(train_images_dir: str | Path) -> pd.DataFrame:
    """
    Compute 64-bit perceptual hash (pHash) for all images across class subfolders.

    Args:
        train_images_dir: Path to the directory containing class subfolders of images.

    Returns:
        DataFrame with columns: file_path, class_name, phash.
        file_path is relative to the project root with forward slashes.
    """
    images_dir = Path(train_images_dir)
    if not images_dir.exists():
        raise FileNotFoundError(f"Training images directory not found: {images_dir}")

    class_dirs = sorted([d for d in images_dir.iterdir() if d.is_dir()])
    records: list[dict[str, str]] = []

    for class_dir in class_dirs:
        class_name = class_dir.name
        image_files = sorted([f for f in class_dir.iterdir() if f.is_file()])
        for img_file in image_files:
            try:
                with Image.open(img_file) as img:
                    h = imagehash.phash(img)
            except Exception as err:
                raise RuntimeError(f"Cannot hash {img_file}") from err

            records.append({
                "file_path": _to_posix_rel_path(img_file),
                "class_name": class_name,
                "phash": str(h),
            })

    return pd.DataFrame(records, columns=["file_path", "class_name", "phash"])


def build_manifest(df: pd.DataFrame) -> pd.DataFrame:
    """
    Build a deduplication manifest DataFrame from precomputed perceptual hashes.

    This is a pure function operating entirely in memory with no file or image access.

    Why exact equality suffices:
    -----------------------------
    A Hamming distance of 0 between two 64-bit perceptual hashes (pHash) indicates
    identical hash strings (exact bitwise equality: hash_a == hash_b). Because equality
    is an equivalence relation (reflexive, symmetric, and transitive), images that
    share an identical pHash form disjoint equivalence classes (clusters). Unlike higher
    distance thresholds (e.g. Hamming distance <= 8) which require pairwise O(N^2)
    searches and graph connected components to resolve non-transitive similarities,
    Hamming distance 0 duplicates can be discovered strictly and efficiently in O(N)
    time via direct hash grouping (groupby 'phash').

    Deduplication rules:
    ---------------------
    1. Singleton:
       Images with a unique pHash (cluster size == 1) are kept:
       status = "keep", reason = "", cluster_id = pd.NA.
    2. Same-Class Cluster:
       When all images in a duplicate cluster share the same class_name:
       Keep exactly one representative: the image with the lowest NUMERIC filename stem
       (e.g., 99999.jpg is kept over 100169.jpg).
       - Kept image: status = "keep", reason = "", cluster_id = <id>
       - Other images: status = "remove", reason = "duplicate", cluster_id = <id>
    3. Cross-Class Cluster:
       When images in a duplicate cluster span more than one class_name, this
       represents contradictory label noise (the same image labeled under different
       categories). ALL members of the cluster are marked for removal:
       - All images: status = "remove", reason = "cross_class_label_conflict", cluster_id = <id>

    Args:
        df: DataFrame containing at least 'file_path', 'class_name', and 'phash'.

    Returns:
        DataFrame with columns: file_path, class_name, status, reason, cluster_id.
        Row ordering preserves the input DataFrame order.
    """
    if df.empty:
        return pd.DataFrame(columns=["file_path", "class_name", "status", "reason", "cluster_id"])

    df_work = df.copy()
    df_work["_orig_idx"] = range(len(df_work))

    records: list[dict[str, Any]] = []
    cluster_id_counter = 1

    grouped = df_work.groupby("phash", sort=False)

    for _, group in grouped:
        if len(group) == 1:
            row = group.iloc[0]
            records.append({
                "_orig_idx": row["_orig_idx"],
                "file_path": row["file_path"],
                "class_name": row["class_name"],
                "status": "keep",
                "reason": "",
                "cluster_id": pd.NA,
            })
        else:
            unique_classes = group["class_name"].unique()
            if len(unique_classes) > 1:
                for _, row in group.iterrows():
                    records.append({
                        "_orig_idx": row["_orig_idx"],
                        "file_path": row["file_path"],
                        "class_name": row["class_name"],
                        "status": "remove",
                        "reason": "cross_class_label_conflict",
                        "cluster_id": cluster_id_counter,
                    })
            else:
                sorted_indices = sorted(
                    group.index,
                    key=lambda idx: _extract_numeric_key(group.loc[idx, "file_path"]),
                )
                kept_idx = sorted_indices[0]
                for idx in group.index:
                    row = group.loc[idx]
                    if idx == kept_idx:
                        records.append({
                            "_orig_idx": row["_orig_idx"],
                            "file_path": row["file_path"],
                            "class_name": row["class_name"],
                            "status": "keep",
                            "reason": "",
                            "cluster_id": cluster_id_counter,
                        })
                    else:
                        records.append({
                            "_orig_idx": row["_orig_idx"],
                            "file_path": row["file_path"],
                            "class_name": row["class_name"],
                            "status": "remove",
                            "reason": "duplicate",
                            "cluster_id": cluster_id_counter,
                        })
            cluster_id_counter += 1

    manifest_df = (
        pd.DataFrame(records)
        .sort_values("_orig_idx")
        .drop(columns=["_orig_idx"])
        .reset_index(drop=True)
    )
    manifest_df["cluster_id"] = manifest_df["cluster_id"].astype("Int64")
    return manifest_df[["file_path", "class_name", "status", "reason", "cluster_id"]]


def main() -> None:
    parser = argparse.ArgumentParser(description="Paddy Doctor duplicate detection and manifest generator.")
    parser.add_argument(
        "--data-dir",
        default="data/raw/paddy-disease-classification/train_images",
        help="Path to training images directory (default: data/raw/paddy-disease-classification/train_images)",
    )
    parser.add_argument(
        "--out",
        default="data/processed/dedup_manifest.csv",
        help="Output CSV path for deduplication manifest (default: data/processed/dedup_manifest.csv)",
    )
    args = parser.parse_args()

    print(f"Scanning images in: {args.data_dir}")
    df_hashes = compute_phash_table(args.data_dir)
    print(f"Computed pHash table for {len(df_hashes):,} images.")

    print("Building deduplication manifest...")
    manifest_df = build_manifest(df_hashes)

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    manifest_df.to_csv(out_path, index=False)
    print(f"Manifest written to: {out_path}")

    # Summary statistics
    total_images = len(manifest_df)
    kept_count = (manifest_df["status"] == "keep").sum()
    removed_count = (manifest_df["status"] == "remove").sum()
    rem_dup = (manifest_df["reason"] == "duplicate").sum()
    rem_conflict = (manifest_df["reason"] == "cross_class_label_conflict").sum()

    print("\n" + "=" * 60)
    print("DEDUPLICATION SUMMARY")
    print("=" * 60)
    print(f"Total images:                         {total_images:>6,}")
    print(f"Total kept:                           {kept_count:>6,} ({kept_count / total_images * 100:.2f}%)")
    print(f"Total removed:                        {removed_count:>6,} ({removed_count / total_images * 100:.2f}%)")
    print(f"  - duplicate:                        {rem_dup:>6,}")
    print(f"  - cross_class_label_conflict:       {rem_conflict:>6,}")

    print("\n" + "=" * 60)
    print("REMOVALS PER CLASS")
    print("=" * 60)
    class_orig = df_hashes["class_name"].value_counts()
    class_rem = (
        manifest_df[manifest_df["status"] == "remove"]["class_name"]
        .value_counts()
        .reindex(class_orig.index, fill_value=0)
    )
    for cls_name, rem_cnt in class_rem.items():
        orig_cnt = class_orig[cls_name]
        pct = (rem_cnt / orig_cnt * 100) if orig_cnt > 0 else 0.0
        print(f"  {cls_name:<30} {rem_cnt:>4} removed / {orig_cnt:>5} ({pct:>5.1f}%)")


if __name__ == "__main__":
    main()
