"""Importing labels made outside UnitRefine (Phy, SortingView, SpikeInterface curation, csv)."""

import json

import pandas as pd
import pytest
import spikeinterface.full as si

from conftest import make_analyzer
from unitrefine.import_labels import import_labels, read_label_file


@pytest.fixture(scope="module")
def integer_id_recording(tmp_path_factory):
    """A recording whose unit ids are integers, as from Kilosort (the session recordings have text ids)."""
    return make_analyzer(tmp_path_factory.mktemp("integer_ids") / "analyzer", num_units=8, seed=2, integer_ids=True)


def imported(label_file, analyzer_folder, save_folder, existing_labels=None):
    labels = import_labels(label_file, str(analyzer_folder), save_folder, existing_labels=existing_labels)
    saved = pd.read_csv(save_folder / "labels.csv")
    return labels, dict(zip(saved["unit_id"].astype(str), saved["quality"]))


def test_phy_cluster_group(tmp_path, recordings):
    label_file = tmp_path / "cluster_group.tsv"
    pd.DataFrame({"cluster_id": [0, 1, 2, 3, 4], "group": ["good", "mua", "noise", "unsorted", "MUA"]}).to_csv(
        label_file, sep="\t", index=False)
    _, saved = imported(label_file, recordings["training"], tmp_path)
    # mua -> MUA; unsorted is not a UnitRefine label and is skipped
    assert saved == {"0": "good", "1": "MUA", "2": "noise", "4": "MUA"}
    # the metrics of the labelled units are saved with them, for training
    assert len(pd.read_csv(tmp_path / "labelled_metrics.csv")) == 4
    assert len(pd.read_csv(tmp_path / "all_metrics.csv")) == 30


def test_phy_cluster_info_uses_the_manual_group_not_kilosort(tmp_path, recordings):
    label_file = tmp_path / "cluster_info.tsv"
    pd.DataFrame({"cluster_id": [0, 1], "KSLabel": ["mua", "mua"], "group": ["good", "noise"]}).to_csv(
        label_file, sep="\t", index=False)
    _, saved = imported(label_file, recordings["training"], tmp_path)
    assert saved == {"0": "good", "1": "noise"}


def test_csv_with_unit_id_and_quality(tmp_path, recordings):
    label_file = tmp_path / "labels.csv"
    pd.DataFrame({"unit_id": [5, 6], "quality": ["noise", "good"]}).to_csv(label_file, index=False)
    _, saved = imported(label_file, recordings["training"], tmp_path)
    assert saved == {"5": "noise", "6": "good"}


def test_sortingview_json(tmp_path, recordings):
    label_file = tmp_path / "sortingview.json"
    label_file.write_text(json.dumps({"labelsByUnit": {"1": ["accept", "mua"], "2": ["noise"]}}))
    _, saved = imported(label_file, recordings["training"], tmp_path)
    assert saved == {"1": "MUA", "2": "noise"}


@pytest.mark.parametrize("version", [1, 2])
def test_spikeinterface_curation_json(tmp_path, recordings, version):
    rows = [{"unit_id": "3", "quality": ["good"]}, {"unit_id": "4", "quality": ["MUA"]}]
    if version == 2:
        rows = [{"unit_id": row["unit_id"], "labels": {"quality": row["quality"]}} for row in rows]
    label_file = tmp_path / "curation.json"
    label_file.write_text(json.dumps({"format_version": str(version), "manual_labels": rows}))
    _, saved = imported(label_file, recordings["training"], tmp_path)
    assert saved == {"3": "good", "4": "MUA"}


def test_integer_unit_ids(tmp_path, integer_id_recording):
    label_file = tmp_path / "cluster_group.tsv"
    pd.DataFrame({"cluster_id": [0, 7], "group": ["good", "noise"]}).to_csv(label_file, sep="\t", index=False)
    _, saved = imported(label_file, integer_id_recording, tmp_path)
    assert saved == {"0": "good", "7": "noise"}


def test_importing_again_keeps_the_other_labels(tmp_path, recordings):
    first = tmp_path / "first.csv"
    pd.DataFrame({"unit_id": [0, 1], "quality": ["good", "good"]}).to_csv(first, index=False)
    imported(first, recordings["training"], tmp_path)
    second = tmp_path / "second.csv"
    pd.DataFrame({"unit_id": [1, 2], "quality": ["noise", "MUA"]}).to_csv(second, index=False)
    existing = pd.read_csv(tmp_path / "labels.csv")
    _, saved = imported(second, recordings["training"], tmp_path, existing_labels=existing)
    # unit 1 gets the new label; unit 0 keeps its label
    assert saved == {"0": "good", "1": "noise", "2": "MUA"}


def test_labels_for_units_not_in_the_recording_are_skipped(tmp_path, recordings):
    label_file = tmp_path / "labels.csv"
    pd.DataFrame({"unit_id": [0, 999], "quality": ["good", "noise"]}).to_csv(label_file, index=False)
    _, saved = imported(label_file, recordings["training"], tmp_path)
    assert saved == {"0": "good"}


def test_unrecognised_file_is_refused(tmp_path):
    label_file = tmp_path / "other.csv"
    pd.DataFrame({"a": [1], "b": [2]}).to_csv(label_file, index=False)
    with pytest.raises(ValueError):
        read_label_file(label_file)
