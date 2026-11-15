"""Tests for train/val/test/stress dataset splitting and metadata loading."""

import pandas as pd
import pytest
from data_prep.split import build_split, load_kept


def _create_synthetic_split_df(n_per_class: int = 100) -> pd.DataFrame:
    """Helper to create a synthetic DataFrame with multiple classes and varieties."""
    rows = []
    classes = ["blast", "brown_spot", "normal"]
    file_id = 100000

    for cls in classes:
        for i in range(n_per_class):
            # 10% Ponni (holdout), remainder ADT45
            variety = "Ponni" if i < 10 else "ADT45"
            rows.append({
                "file_path": f"data/raw/train_images/{cls}/{file_id}.jpg",
                "image_id": f"{file_id}.jpg",
                "label": cls,
                "variety": variety,
                "age": 50,
            })
            file_id += 1

    return pd.DataFrame(rows)


def test_split_partition_and_completeness():
    """No file_path appears in more than one split, and every input row appears exactly once."""
    df = _create_synthetic_split_df(n_per_class=50)
    result = build_split(df, holdout_variety="Ponni", val_frac=0.15, test_frac=0.15, seed=42)

    assert len(result) == len(df)
    assert set(result["file_path"]) == set(df["file_path"])
    assert result["split"].isin(["train", "val", "test", "stress"]).all()

    # Disjointness across splits
    grouped_paths = result.groupby("split")["file_path"].apply(set)
    all_seen = set()
    for s_name, path_set in grouped_paths.items():
        assert all_seen.isdisjoint(path_set), f"Overlap detected in split {s_name}"
        all_seen.update(path_set)


def test_holdout_variety_is_stress():
    """Every row of the holdout variety is stress, and no other row is."""
    df = _create_synthetic_split_df(n_per_class=40)
    result = build_split(df, holdout_variety="Ponni", val_frac=0.15, test_frac=0.15, seed=42)

    ponni_mask = result["variety"] == "Ponni"
    assert (result.loc[ponni_mask, "split"] == "stress").all()
    assert (result.loc[~ponni_mask, "split"] != "stress").all()


def test_stratified_split_proportions():
    """For each class, train, val and test counts are within 2 images of 70/15/15 of non-holdout total."""
    # 100 per class: 10 Ponni (stress) and 90 non-stress per class
    # 90 * 0.70 = 63 train; 90 * 0.15 = 13.5 val; 90 * 0.15 = 13.5 test
    df = _create_synthetic_split_df(n_per_class=100)
    result = build_split(df, holdout_variety="Ponni", val_frac=0.15, test_frac=0.15, seed=42)

    non_stress = result[result["split"] != "stress"]
    for cls in df["label"].unique():
        sub = non_stress[non_stress["label"] == cls]
        total_cls = len(sub)  # 90
        n_train = (sub["split"] == "train").sum()
        n_val = (sub["split"] == "val").sum()
        n_test = (sub["split"] == "test").sum()

        expected_train = total_cls * 0.70
        expected_val = total_cls * 0.15
        expected_test = total_cls * 0.15

        assert abs(n_train - expected_train) <= 2, f"Train mismatch for {cls}: {n_train} vs {expected_train}"
        assert abs(n_val - expected_val) <= 2, f"Val mismatch for {cls}: {n_val} vs {expected_val}"
        assert abs(n_test - expected_test) <= 2, f"Test mismatch for {cls}: {n_test} vs {expected_test}"


def test_seed_reproducibility():
    """The same seed gives an identical split, and a different seed gives a different one."""
    df = _create_synthetic_split_df(n_per_class=50)

    split_seed42_a = build_split(df, seed=42)
    split_seed42_b = build_split(df, seed=42)
    split_seed99 = build_split(df, seed=99)

    assert list(split_seed42_a["split"]) == list(split_seed42_b["split"])
    assert list(split_seed42_a["split"]) != list(split_seed99["split"])


def test_load_kept_filters_non_keep():
    """load_kept drops every row whose status is not keep."""
    manifest_df = pd.DataFrame([
        {"file_path": "images/blast/100.jpg", "class_name": "blast", "status": "keep"},
        {"file_path": "images/blast/101.jpg", "class_name": "blast", "status": "remove"},
        {"file_path": "images/normal/102.jpg", "class_name": "normal", "status": "keep"},
    ])
    train_csv_df = pd.DataFrame([
        {"image_id": "100.jpg", "label": "blast", "variety": "ADT45", "age": 50},
        {"image_id": "101.jpg", "label": "blast", "variety": "ADT45", "age": 50},
        {"image_id": "102.jpg", "label": "normal", "variety": "Ponni", "age": 60},
    ])

    kept = load_kept(manifest_df, train_csv_df)

    assert len(kept) == 2
    assert set(kept["image_id"]) == {"100.jpg", "102.jpg"}
    assert "101.jpg" not in kept["image_id"].values


def test_load_kept_missing_csv_row_raises():
    """load_kept raises ValueError when a kept image has no CSV row."""
    manifest_df = pd.DataFrame([
        {"file_path": "images/blast/100.jpg", "class_name": "blast", "status": "keep"},
        {"file_path": "images/blast/999.jpg", "class_name": "blast", "status": "keep"},
    ])
    train_csv_df = pd.DataFrame([
        {"image_id": "100.jpg", "label": "blast", "variety": "ADT45", "age": 50},
    ])

    with pytest.raises(ValueError, match="no matching CSV"):
        load_kept(manifest_df, train_csv_df)


def test_load_kept_label_mismatch_raises():
    """load_kept raises ValueError when the CSV label disagrees with the class folder name."""
    manifest_df = pd.DataFrame([
        {"file_path": "images/blast/100.jpg", "class_name": "blast", "status": "keep"},
    ])
    train_csv_df = pd.DataFrame([
        {"image_id": "100.jpg", "label": "tungro", "variety": "ADT45", "age": 50},
    ])

    with pytest.raises(ValueError, match="label mismatch"):
        load_kept(manifest_df, train_csv_df)


def test_load_kept_duplicate_image_id_raises():
    """load_kept raises ValueError when train_csv_df contains duplicate image_id."""
    manifest_df = pd.DataFrame([
        {"file_path": "images/blast/100.jpg", "class_name": "blast", "status": "keep"},
    ])
    train_csv_df = pd.DataFrame([
        {"image_id": "100.jpg", "label": "blast", "variety": "ADT45", "age": 50},
        {"image_id": "100.jpg", "label": "blast", "variety": "ADT45", "age": 50},
    ])

    with pytest.raises(ValueError, match="Duplicate image_id"):
        load_kept(manifest_df, train_csv_df)
