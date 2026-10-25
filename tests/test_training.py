"""Training models in the Train window, and applying them to a new recording, separately and as a cascade."""

import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import spikeinterface.full as si
from sklearn.metrics import balanced_accuracy_score

from unitrefine.evaluation import normalise_path, read_trained_on
from unitrefine.predictions import (
    get_model_classes,
    get_predictions,
    get_saved_or_new_predictions,
    predictions_are_up_to_date,
)

TASKS = {
    # task: (classes, number of units trained on: all 30, or only good and MUA for SUA vs MUA)
    "noise_vs_neural": ({"neural", "noise"}, 30),
    "sua_vs_mua": ({"good", "MUA"}, 20),
    "sua_vs_all": ({"good", "not-sua"}, 30),
}


def model_folder(project, task):
    return Path(project) / "models" / f"test_model_{task}"


@pytest.mark.parametrize("task", TASKS)
def test_trained_model_has_the_task_classes(trained_project, task):
    assert set(get_model_classes(model_folder(trained_project, task))) == TASKS[task][0]


@pytest.mark.parametrize("task", TASKS)
def test_saved_test_set_reproduces_the_reported_accuracy(trained_project, task):
    """The test set is rebuilt after training; it is only saved if it gives train_model's own score."""
    folder = model_folder(trained_project, task)
    test = pd.read_csv(folder / "test_predictions.csv")
    reported = pd.read_csv(folder / "model_accuracies.csv")["balanced_accuracy"].values[0]
    assert balanced_accuracy_score(test["human"], test["predicted"]) == pytest.approx(reported, abs=6e-5)


@pytest.mark.parametrize("task", TASKS)
def test_the_units_trained_on_are_recorded(trained_project, recordings, task):
    folder = model_folder(trained_project, task)
    units, status = read_trained_on(folder)
    assert status == "recorded"
    assert len(units[normalise_path(recordings["training"])]) == TASKS[task][1]

    # each row of the training data is matched to its analyzer and unit (used to replace relabelled rows)
    training_units = pd.read_csv(folder / "training_units.csv", dtype=str)
    training_data = pd.read_csv(folder / "training_data.csv", index_col="unit_id")
    assert list(training_units["unit_id"]) == [str(unit_id) for unit_id in training_data.index]
    assert set(training_units["analyzer"]) == {str(recordings["training"])}


def test_applying_a_model_does_not_change_the_analyzer(trained_project, recordings):
    """The predictions are saved in UnitRefine's csv files: nothing is written to the analyzer, which can be on s3,
    where it can't be written to."""
    analyzer = si.load_sorting_analyzer(recordings["new"], load_extensions=False)
    properties = set(analyzer.sorting.get_property_keys())
    get_predictions(analyzer, "binary", [model_folder(trained_project, "sua_vs_all")])
    assert set(analyzer.sorting.get_property_keys()) == properties


def test_a_model_applies_to_a_new_recording(trained_project, recordings):
    analyzer = si.load_sorting_analyzer(recordings["new"], load_extensions=False)
    folder = model_folder(trained_project, "sua_vs_all")
    labels, confidences, selection_confidence = get_predictions(analyzer, "binary", [folder])
    predicted = labels[f"{folder.name}_model_label"]
    assert len(predicted) == 25
    assert set(predicted) <= {"good", "not-sua"}
    confidence = confidences[f"{folder.name}_confidence"]
    # the confidence is the probability of the predicted class: at least 0.5 with two classes
    assert np.all((confidence >= 0.5) & (confidence <= 1))
    assert np.array_equal(selection_confidence, confidence)


def test_a_cascade_labels_neural_units_with_layer_2(trained_project, recordings):
    analyzer = si.load_sorting_analyzer(recordings["new"], load_extensions=False)
    layer1, layer2 = model_folder(trained_project, "noise_vs_neural"), model_folder(trained_project, "sua_vs_mua")
    labels, confidences, selection_confidence = get_predictions(analyzer, "cascade", [layer1, layer2])

    cascade = labels["cascade_model_label"]
    layer1_labels = labels[f"L1_{layer1.name}_model_label"]
    layer2_labels = labels[f"L2_{layer2.name}_model_label"]
    assert set(cascade) <= {"good", "MUA", "noise"}
    # layer 2 only labels the units layer 1 calls neural; the others are noise
    neural = layer1_labels[layer1_labels == "neural"].index
    assert list(layer2_labels.index) == list(neural)
    assert (cascade[layer1_labels == "noise"] == "noise").all()
    assert (cascade[neural] == layer2_labels[neural]).all()
    # confidence of the cascade: p(layer 1) x p(layer 2) for neural units, p(layer 1) for noise
    p1 = confidences[f"L1_{layer1.name}_confidence"]
    p2 = confidences[f"L2_{layer2.name}_confidence"]
    is_noise = (layer1_labels == "noise").reindex(analyzer.unit_ids).values
    assert np.allclose(selection_confidence, np.where(is_noise, p1, p1 * p2))


def test_saved_predictions_are_reused(trained_project, recordings, tmp_path):
    analyzer = si.load_sorting_analyzer(recordings["new"], load_extensions=False)
    folder = model_folder(trained_project, "sua_vs_all")
    path = tmp_path / "predictions.csv"
    first = get_saved_or_new_predictions(analyzer, str(recordings["new"]), "binary", [folder], path)
    assert predictions_are_up_to_date(path, [folder], str(recordings["new"]))
    second = get_saved_or_new_predictions(analyzer, str(recordings["new"]), "binary", [folder], path)
    for column in first[0]:
        assert list(first[0][column].astype(str)) == list(second[0][column].astype(str))
    for column in first[1]:
        assert np.allclose(first[1][column], second[1][column])
