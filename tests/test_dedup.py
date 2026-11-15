"""Tests for duplicate detection manifest generation."""

import pandas as pd
import pytest
from data_prep.dedup import build_manifest


def test_same_class_triple_lowest_numeric_stem():
    """
    A same-class triple sharing an identical hash must keep exactly one image:
    the one with the lowest numeric filename stem.

    Includes 99999.jpg vs 100169.jpg where numeric and alphabetical order differ
    ('100169' < '99999' alphabetically, but 99999 < 100169 numerically).
    """
    df = pd.DataFrame([
        {"file_path": "images/blast/100169.jpg", "class_name": "blast", "phash": "hash1"},
        {"file_path": "images/blast/99999.jpg", "class_name": "blast", "phash": "hash1"},
        {"file_path": "images/blast/100500.jpg", "class_name": "blast", "phash": "hash1"},
    ])

    manifest = build_manifest(df)

    assert len(manifest) == 3
    # 99999.jpg must be kept
    row_99999 = manifest[manifest["file_path"] == "images/blast/99999.jpg"].iloc[0]
    assert row_99999["status"] == "keep"
    assert row_99999["reason"] == ""
    assert pd.notna(row_99999["cluster_id"])

    # 100169.jpg and 100500.jpg must be marked remove with duplicate
    row_100169 = manifest[manifest["file_path"] == "images/blast/100169.jpg"].iloc[0]
    assert row_100169["status"] == "remove"
    assert row_100169["reason"] == "duplicate"
    assert row_100169["cluster_id"] == row_99999["cluster_id"]

    row_100500 = manifest[manifest["file_path"] == "images/blast/100500.jpg"].iloc[0]
    assert row_100500["status"] == "remove"
    assert row_100500["reason"] == "duplicate"
    assert row_100500["cluster_id"] == row_99999["cluster_id"]

    # Exactly one kept
    assert (manifest["status"] == "keep").sum() == 1
    assert (manifest["status"] == "remove").sum() == 2


def test_cross_class_pair_conflict():
    """A cross-class pair sharing an identical hash has both images removed with conflict reason."""
    df = pd.DataFrame([
        {"file_path": "images/blast/105300.jpg", "class_name": "blast", "phash": "conflict_hash"},
        {"file_path": "images/downy_mildew/100821.jpg", "class_name": "downy_mildew", "phash": "conflict_hash"},
    ])

    manifest = build_manifest(df)

    assert len(manifest) == 2
    assert (manifest["status"] == "remove").all()
    assert (manifest["reason"] == "cross_class_label_conflict").all()
    # Both share the same cluster_id
    assert pd.notna(manifest["cluster_id"].iloc[0])
    assert manifest["cluster_id"].iloc[0] == manifest["cluster_id"].iloc[1]


def test_cross_class_triple_conflict():
    """A 3-image cluster spanning two classes has all three removed as conflicts."""
    df = pd.DataFrame([
        {"file_path": "images/hispa/107223.jpg", "class_name": "hispa", "phash": "triple_hash"},
        {"file_path": "images/hispa/107224.jpg", "class_name": "hispa", "phash": "triple_hash"},
        {"file_path": "images/tungro/109617.jpg", "class_name": "tungro", "phash": "triple_hash"},
    ])

    manifest = build_manifest(df)

    assert len(manifest) == 3
    assert (manifest["status"] == "remove").all()
    assert (manifest["reason"] == "cross_class_label_conflict").all()
    assert manifest["cluster_id"].nunique() == 1
    assert pd.notna(manifest["cluster_id"].iloc[0])


def test_singletons_kept_and_length_preserved():
    """Singletons are all kept, blank reason, blank cluster_id, exactly one row per input row."""
    df = pd.DataFrame([
        {"file_path": "images/normal/100001.jpg", "class_name": "normal", "phash": "hash_a"},
        {"file_path": "images/blast/100002.jpg", "class_name": "blast", "phash": "hash_b"},
        {"file_path": "images/hispa/100003.jpg", "class_name": "hispa", "phash": "hash_c"},
    ])

    manifest = build_manifest(df)

    assert len(manifest) == len(df)
    assert list(manifest["file_path"]) == list(df["file_path"])
    assert list(manifest["class_name"]) == list(df["class_name"])
    assert (manifest["status"] == "keep").all()
    assert (manifest["reason"] == "").all()
    assert manifest["cluster_id"].isna().all()


def test_build_manifest_preserves_input_row_order():
    """
    Test that build_manifest preserves input row order when the input mixes
    singletons, a same-class duplicate pair, and a cross-class pair.
    """
    df = pd.DataFrame([
        {"file_path": "images/normal/100001.jpg", "class_name": "normal", "phash": "hash_singleton_1"},
        {"file_path": "images/blast/100050.jpg", "class_name": "blast", "phash": "hash_same_class"},
        {"file_path": "images/hispa/100002.jpg", "class_name": "hispa", "phash": "hash_singleton_2"},
        {"file_path": "images/tungro/100010.jpg", "class_name": "tungro", "phash": "hash_cross_class"},
        {"file_path": "images/blast/100020.jpg", "class_name": "blast", "phash": "hash_same_class"},
        {"file_path": "images/downy_mildew/100011.jpg", "class_name": "downy_mildew", "phash": "hash_cross_class"},
        {"file_path": "images/dead_heart/100003.jpg", "class_name": "dead_heart", "phash": "hash_singleton_3"},
    ])

    manifest = build_manifest(df)

    assert list(manifest["file_path"]) == list(df["file_path"])
