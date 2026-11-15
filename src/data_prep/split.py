"""
Dataset splitting module for Paddy Doctor disease classification.

Performs stratified train/val/test splits and isolates holdout variety stress sets.
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd
from sklearn.model_selection import train_test_split


def build_split(
    df: pd.DataFrame,
    holdout_variety: str = "Ponni",
    val_frac: float = 0.15,
    test_frac: float = 0.15,
    seed: int = 42,
) -> pd.DataFrame:
    """
    Split dataset into train, val, test, and stress subsets.

    Why a whole variety is held out:
    --------------------------------
    Exploratory metadata analysis reveals strong confounding between rice varieties
    and disease categories (e.g. normal appears in only 4 of 10 varieties; 41 of 100
    label-by-variety pairs are empty; small varieties contain only one or two disease
    types). In deployment on edge devices, visual leaf characteristics inherent to
    specific varieties (leaf elongation, texture, baseline green tint) can act as
    spurious shortcuts for disease classification if learned by the network.

    Holding out an entire variety ('Ponni') as an out-of-distribution stress test set
    evaluates whether the trained model genuinely recognizes disease pathologies or
    merely memorized variety-correlated visual cues, providing an uncorrupted measure
    of cross-variety generalization.

    Deduplication and Stratification Rules:
    ---------------------------------------
    1. Rows matching `holdout_variety` are assigned to the 'stress' split.
    2. All other rows are split via a two-stage stratified partition on 'label':
       - Stage 1: Separate 'train' from remaining rows (val + test).
       - Stage 2: Partition remaining rows equally into 'val' and 'test'.
    3. Preserves original DataFrame row ordering.

    Args:
        df: Input DataFrame containing at least 'variety' and 'label'.
        holdout_variety: Rice variety name to isolate as the stress set.
        val_frac: Fraction of non-holdout data for validation (default: 0.15).
        test_frac: Fraction of non-holdout data for testing (default: 0.15).
        seed: Random state seed for reproducibility.

    Returns:
        DataFrame with an added 'split' column ('train', 'val', 'test', 'stress').
    """
    df_out = df.copy()
    df_out["split"] = ""

    # 1. Identify holdout variety as stress set
    is_stress = df_out["variety"] == holdout_variety
    df_out.loc[is_stress, "split"] = "stress"

    non_stress_indices = df_out[~is_stress].index
    non_stress_labels = df_out.loc[non_stress_indices, "label"]

    # Total holdout fraction for val + test
    rest_frac = val_frac + test_frac
    if rest_frac <= 0 or rest_frac >= 1.0:
        raise ValueError(f"Combined val_frac + test_frac must be in (0, 1), got {rest_frac}")

    # Stage 1: Split non-stress into train vs (val + test)
    train_idx, rest_idx = train_test_split(
        non_stress_indices,
        test_size=rest_frac,
        stratify=non_stress_labels,
        random_state=seed,
    )

    # Stage 2: Split rest equally into val vs test
    rest_labels = df_out.loc[rest_idx, "label"]
    test_ratio_of_rest = test_frac / rest_frac

    val_idx, test_idx = train_test_split(
        rest_idx,
        test_size=test_ratio_of_rest,
        stratify=rest_labels,
        random_state=seed,
    )

    df_out.loc[train_idx, "split"] = "train"
    df_out.loc[val_idx, "split"] = "val"
    df_out.loc[test_idx, "split"] = "test"

    return df_out


def load_kept(manifest_df: pd.DataFrame, train_csv_df: pd.DataFrame) -> pd.DataFrame:
    """
    Filter deduplication manifest for kept images and join with CSV metadata.

    Validates:
    1. Every kept image file has a corresponding row in train_csv_df.
    2. No duplicate image_id values exist in train_csv_df.
    3. The disease label in train_csv_df strictly matches the manifest class folder.

    Args:
        manifest_df: Deduplication manifest containing 'file_path', 'class_name', 'status'.
        train_csv_df: Metadata DataFrame containing 'image_id', 'label', 'variety', 'age'.

    Returns:
        Joined DataFrame with columns from both sources for kept images.
    """
    # 1. Filter status == "keep"
    kept = manifest_df[manifest_df["status"] == "keep"].copy()

    # 2. Check for duplicate image_id in CSV metadata
    if train_csv_df["image_id"].duplicated().any():
        dups = train_csv_df[train_csv_df["image_id"].duplicated()]["image_id"].tolist()
        raise ValueError(f"Duplicate image_id found in CSV metadata: {dups[:5]}")

    # 3. Extract filename stem and verify all kept images exist in CSV
    kept["_filename"] = kept["file_path"].apply(lambda p: Path(p).name)
    csv_ids = set(train_csv_df["image_id"])
    missing_ids = [fid for fid in kept["_filename"] if fid not in csv_ids]
    if missing_ids:
        raise ValueError(
            f"Found {len(missing_ids)} kept image(s) with no matching CSV metadata row: {missing_ids[:5]}"
        )

    # 4. Join on filename == image_id
    merged = kept.merge(
        train_csv_df,
        left_on="_filename",
        right_on="image_id",
        how="inner",
    ).drop(columns=["_filename"])

    # 5. Check for label mismatches between class folder and CSV label
    mismatches = merged[merged["class_name"] != merged["label"]]
    if not mismatches.empty:
        details = [
            f"  {row['file_path']} -> folder: '{row['class_name']}', csv: '{row['label']}'"
            for _, row in mismatches.iterrows()
        ]
        raise ValueError(
            f"Found {len(mismatches)} label mismatch(es) between folder name and CSV label:\n"
            + "\n".join(details)
        )

    return merged


def main() -> None:
    manifest_path = Path("data/processed/dedup_manifest.csv")
    csv_path = Path("data/raw/paddy-disease-classification/train.csv")
    out_path = Path("data/processed/split.csv")

    print(f"Loading manifest: {manifest_path}")
    manifest_df = pd.read_csv(manifest_path)

    print(f"Loading CSV metadata: {csv_path}")
    train_csv_df = pd.read_csv(csv_path)

    print("\nFirst 5 image_id values before joining:")
    for img_id in train_csv_df["image_id"].head(5):
        print(f"  {img_id}")

    print("\nValidating and joining kept records...")
    kept_df = load_kept(manifest_df, train_csv_df)
    print(f"Kept images matched: {len(kept_df):,}")

    print("Building stratified splits with 'Ponni' variety holdout...")
    split_df = build_split(kept_df, holdout_variety="Ponni", val_frac=0.15, test_frac=0.15, seed=42)

    export_cols = ["file_path", "image_id", "label", "variety", "age", "split"]
    out_path.parent.mkdir(parents=True, exist_ok=True)
    split_df[export_cols].to_csv(out_path, index=False)
    print(f"Split manifest written to: {out_path} ({len(split_df):,} rows)")

    # 1. Number of images per split
    print("\n" + "=" * 60)
    print("IMAGES PER SPLIT")
    print("=" * 60)
    split_counts = split_df["split"].value_counts()
    for s_name, count in split_counts.items():
        pct = (count / len(split_df)) * 100
        print(f"  {s_name:<10} {count:>5,} images ({pct:>5.1f}%)")

    # 2. Split x Label table with counts and percentages
    print("\n" + "=" * 60)
    print("SPLIT x LABEL COUNTS & PERCENTAGES")
    print("=" * 60)
    split_label_ct = pd.crosstab(split_df["label"], split_df["split"])
    split_label_pct = pd.crosstab(split_df["label"], split_df["split"], normalize="index") * 100
    combined_table = pd.DataFrame()
    for col in ["train", "val", "test", "stress"]:
        if col in split_label_ct.columns:
            combined_table[f"{col}_count"] = split_label_ct[col]
            combined_table[f"{col}_%"] = split_label_pct[col].round(1)
        else:
            combined_table[f"{col}_count"] = 0
            combined_table[f"{col}_%"] = 0.0
    print(combined_table.to_string())

    # 3. Split x Variety table
    print("\n" + "=" * 60)
    print("SPLIT x VARIETY TABLE")
    print("=" * 60)
    variety_table = pd.crosstab(split_df["variety"], split_df["split"])
    print(variety_table.to_string())

    # 4. Classes present in the stress split
    print("\n" + "=" * 60)
    print("CLASSES PRESENT IN STRESS SPLIT")
    print("=" * 60)
    stress_classes = sorted(split_df[split_df["split"] == "stress"]["label"].unique().tolist())
    all_classes = sorted(split_df["label"].unique().tolist())
    print(f"Total classes in dataset: {len(all_classes)}")
    print(f"Classes in stress split ({len(stress_classes)}):")
    for cls in stress_classes:
        cnt = (split_df[split_df["split"] == "stress"]["label"] == cls).sum()
        print(f"  - {cls:<30} ({cnt} images)")

    missing_in_stress = set(all_classes) - set(stress_classes)
    if missing_in_stress:
        print(f"\nNote: Classes with 0 images in stress split: {sorted(list(missing_in_stress))}")


if __name__ == "__main__":
    main()
