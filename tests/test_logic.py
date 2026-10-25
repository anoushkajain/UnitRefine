"""The calculations behind UnitRefine's results, which don't need recordings or the GUI."""

import numpy as np
import pandas as pd
import pytest
from sklearn.metrics import balanced_accuracy_score, roc_auc_score

from unitrefine import save_labels_or_backup, unwritable_file_message
from unitrefine.predictions import (
    as_model_input,
    map_quality_to_model_class,
    metrics_for_model,
    selection_key,
)


@pytest.mark.parametrize("quality, classes, expected", [
    # noise vs neural: good and MUA are neural (called "good" in models from the Hugging Face Hub)
    ("good", ["neural", "noise"], "neural"),
    ("MUA", ["neural", "noise"], "neural"),
    ("noise", ["neural", "noise"], "noise"),
    ("MUA", ["good", "noise"], "good"),
    # SUA vs MUA: noise is not one of its classes
    ("good", ["MUA", "good"], "good"),
    ("noise", ["MUA", "good"], None),
    ("MUA", ["mua", "sua"], "mua"),
    # SUA vs all
    ("good", ["good", "not-sua"], "good"),
    ("MUA", ["good", "not-sua"], "not-sua"),
    ("noise", ["good", "not-sua"], "not-sua"),
])
def test_map_quality_to_model_class(quality, classes, expected):
    assert map_quality_to_model_class(quality, classes) == expected


def test_selection_key():
    assert selection_key("binary", ["model_a"]) == "model_a"
    assert selection_key("binary", ["model_a", "model_b"]) == "model_a+model_b"
    assert selection_key("cascade", ["layer1", "layer2"]) == "cascade__layer1__layer2"
    # long keys are shortened, so file paths stay short enough for Windows
    long_key = selection_key("binary", [f"a_very_long_model_name_{index}" for index in range(10)])
    assert len(long_key) < 100


def test_metrics_for_model_converts_old_names_like_spikeinterface():
    metrics = pd.DataFrame({
        "peak_to_trough_duration": [1.0, 2.0],
        "peak_after_to_trough_ratio": [0.5, -0.25],
        "trough_half_width": [3.0, 4.0],
        "snr": [5.0, 6.0],
    })
    old_model = {"requirements": {"spikeinterface": "0.102.0"}}
    converted = metrics_for_model(metrics, old_model)
    assert list(converted.columns) == ["peak_to_valley", "peak_trough_ratio", "half_width", "snr"]
    # peak_trough_ratio had the opposite sign
    assert list(converted["peak_trough_ratio"]) == [-0.5, 0.25]
    assert list(converted["snr"]) == [5.0, 6.0]

    # models trained with SpikeInterface 0.103.2 or later use the new names
    new_model = {"requirements": {"spikeinterface": "0.105.3"}}
    assert metrics_for_model(metrics, new_model).equals(metrics)
    assert metrics_for_model(metrics, None).equals(metrics)


def test_metrics_for_model_matches_spikeinterface():
    """The same result as SpikeInterface's own conversion, where this SpikeInterface version has it."""
    from spikeinterface.curation import model_based_curation

    convert = getattr(model_based_curation, "_handle_backwards_compatibility_in_metrics", None)
    if convert is None:
        pytest.skip("this SpikeInterface version has no module-level conversion function")
    rng = np.random.default_rng(0)
    metrics = pd.DataFrame(rng.normal(size=(20, 4)), columns=["peak_to_trough_duration", "peak_after_to_trough_ratio",
                                                               "trough_half_width", "firing_rate"])
    model_info = {"requirements": {"spikeinterface": "0.101.0"}}
    assert metrics_for_model(metrics, model_info).equals(convert(metrics, model_info))


def test_as_model_input_is_float32_without_infinities():
    metrics = pd.DataFrame({"a": [1.0, np.inf, -np.inf, 0.1]})
    converted = as_model_input(metrics)
    assert converted["a"].dtype == np.float32
    assert converted["a"].isna().tolist() == [False, True, True, False]


def test_bootstrap_balanced_accuracy_matches_sklearn():
    from unitrefine.figures import bootstrap_balanced_accuracy

    rng = np.random.default_rng(3)
    true = rng.choice(["good", "MUA", "noise"], 40)
    predicted = rng.choice(["good", "MUA", "noise"], 40)
    ours = bootstrap_balanced_accuracy(true, predicted, num_resamples=100, seed=5)
    resample = np.random.default_rng(5)
    reference = []
    for _ in range(100):
        indices = resample.integers(0, 40, 40)
        reference.append(balanced_accuracy_score(true[indices], predicted[indices]))
    assert np.allclose(ours, reference)


@pytest.mark.parametrize("seed", range(5))
def test_metric_auc_matches_sklearn(seed):
    """The per-metric AUC-ROC of the paper figure, from ranks, is sklearn's roc_auc_score, ties included."""
    from unitrefine.figures import roc_auc_from_ranks

    rng = np.random.default_rng(seed)
    is_positive = rng.random(50) < 0.4
    # rounded, so there are many ties
    values = np.round(rng.normal(size=50), 1)
    assert roc_auc_from_ranks(is_positive, values) == pytest.approx(roc_auc_score(is_positive, values))


def test_metric_auc_per_analyzer():
    """With 3 or more analyzers, the AUC-ROC is one value per analyzer, as in the paper."""
    from unitrefine.figures import metric_auc_values

    rng = np.random.default_rng(0)
    table = pd.DataFrame({
        "analyzer": np.repeat(["a", "b", "c"], 30),
        "label": np.tile(["good", "MUA", "noise"], 30),
        "snr": rng.normal(size=90),
    })
    values, sources, method = metric_auc_values(table, "good", ["snr"])
    assert method == "analyzers"
    assert sources["snr"] == ["a", "b", "c"]
    for analyzer, value in zip(sources["snr"], values["snr"]):
        part = table[table["analyzer"] == analyzer]
        assert value == pytest.approx(roc_auc_score(part["label"] == "good", part["snr"]))


def test_unwritable_file_message_says_how_to_fix_it(tmp_path):
    message = unwritable_file_message(tmp_path / "config.json", PermissionError("denied"))
    assert "config.json" in message and "Read-only" in message


def test_labels_are_backed_up_when_the_file_cant_be_written(tmp_path, monkeypatch, capsys):
    monkeypatch.setattr("pathlib.Path.home", lambda: tmp_path)
    labels = pd.DataFrame({"quality": ["good", "noise"], "unit_id": [1, 2]})

    def fail(table, path):
        raise PermissionError(f"[Errno 13] Permission denied: '{path}'")

    saved_to = save_labels_or_backup(labels, tmp_path / "labels.csv", write=fail, index=False)
    assert saved_to != tmp_path / "labels.csv"
    assert saved_to.parent == tmp_path and saved_to.name.endswith("labels.csv")
    assert pd.read_csv(saved_to).equals(labels)
    assert "UNITREFINE_LABELS_BACKUP:" in capsys.readouterr().out


def test_labels_are_saved_normally(tmp_path):
    labels = pd.DataFrame({"quality": ["good"], "unit_id": [1]})
    assert save_labels_or_backup(labels, tmp_path / "labels.csv", index=False) == tmp_path / "labels.csv"
    assert pd.read_csv(tmp_path / "labels.csv").equals(labels)


def test_analyzer_output_names_are_unique_and_short():
    from unitrefine.predictions import MAX_OUTPUT_NAME_LENGTH, analyzer_output_names

    s3_name = "experiment1_Record Node 101#Neuropix-PXI-100.ProbeA-AP_recording{}.zarr"
    names = analyzer_output_names([
        "C:/data/mouse1/sorting_analyzer", "C:/data/mouse2/sorting_analyzer",
        f"s3://bucket/session/postprocessed/{s3_name.format(1)}", f"s3://bucket/session/postprocessed/{s3_name.format(2)}",
    ])
    # parent folders tell analyzers with the same name apart
    assert names[:2] == ["mouse1_sorting_analyzer", "mouse2_sorting_analyzer"]
    # long names (e.g. on s3) are shortened, so file paths stay short enough for Windows, and stay unique
    assert all(len(name) <= MAX_OUTPUT_NAME_LENGTH for name in names) and len(set(names)) == 4
    assert "experiment1_Record Node 101" in names[2]
