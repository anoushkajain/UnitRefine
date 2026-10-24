"""Active learning (section 4): finding the least confident units, and retraining with the relabelled units."""

from pathlib import Path

import pandas as pd
import pytest

from conftest import targets_of
from unitrefine.evaluation import normalise_path, read_trained_on
from unitrefine.main import MainWindow, load_project
from unitrefine.predictions import find_least_confident_units, relabelled_units_path, units_to_relabel_path
from unitrefine.retraining_history import read_log

MODEL = "test_model_sua_vs_all"


@pytest.fixture
def project(project_copy, trained_project, recordings):
    """The trained project with the new recording added, without labels: analyzer 0 is all labelled, analyzer 1 has none."""
    folder = project_copy(trained_project)
    loaded = load_project(folder)
    loaded.add_analyzer(str(recordings["new"]))
    loaded.save()
    return folder


@pytest.fixture
def window(project, qapp):
    window = MainWindow(load_project(project))
    yield window
    window.close()


def model_folder(project, name=MODEL):
    return Path(project) / "models" / name


def relabel(target, labels, key=MODEL):
    """Saves relabels as the Relabel window does, as {unit id: label}."""
    pd.DataFrame({"quality": list(labels.values()), "unit_id": list(labels)}).to_csv(relabelled_units_path(target, key), index=False)


def labels_by_row(folder):
    """A model's training labels, as {(analyzer, unit id): class}, from training_units.csv and labels.csv."""
    from spikeinterface.curation.model_based_curation import load_model
    from unitrefine.train import read_training_units

    labels = pd.read_csv(Path(folder) / "labels.csv")
    _, model_info = load_model(model_folder=folder, trust_model=True)
    codes = labels.drop(columns="unit_index").iloc[:, 0]
    units = read_training_units(folder, len(labels))
    assert units is not None
    by_row = {(analyzer, unit_id): model_info["label_conversion"][str(code)]
              for analyzer, unit_id, code in zip(units["analyzer"], units["unit_id"], codes)}
    # one row per unit
    assert len(by_row) == len(labels)
    return by_row


def find(project, num_units):
    targets = targets_of(project)
    num_picked, skipped = find_least_confident_units(targets, "binary", [model_folder(project)], MODEL, num_units)
    assert skipped == []
    picked = [pd.read_csv(units_to_relabel_path(target, MODEL), dtype=str)["unit_id"].tolist() for target in targets]
    return targets, num_picked, picked


def test_find_never_picks_labelled_units(project):
    targets, num_picked, picked = find(project, 10)
    # analyzer 0 was labelled for training, so all 10 units are picked from the new analyzer
    assert num_picked == {0: 0, 1: 10}
    assert picked[0] == []
    first_round = picked[1]

    # relabelled units are not picked again (sampling without replacement), whichever label they got
    relabel(targets[1], {unit_id: "good" for unit_id in first_round})
    _, num_picked, picked = find(project, 10)
    assert num_picked[1] == 10
    assert set(picked[1]).isdisjoint(first_round)

    # 25 units: 5 are left after two rounds of 10
    relabel(targets[1], {unit_id: "good" for unit_id in first_round + picked[1]}, key="another_selection")
    _, num_picked, picked = find(project, 10)
    assert num_picked[1] == 5


def test_least_confident_units_are_picked_first(project):
    _, _, picked = find(project, 25)
    ranked = pd.read_csv(units_to_relabel_path(targets_of(project)[1], MODEL))
    assert list(ranked["confidence"]) == sorted(ranked["confidence"])


def test_retraining_adds_the_relabelled_units(window, project):
    targets = window.relabel_targets()
    training_labels = pd.read_csv(Path(targets[0]["save_folder"]) / "labels.csv", dtype=str)
    original = dict(zip(training_labels["unit_id"], training_labels["quality"]))
    by_label = {label: [unit_id for unit_id, quality in original.items() if quality == label] for label in ["good", "MUA", "noise"]}

    # for SUA vs all: good -> noise and noise -> good change the class, MUA -> noise doesn't (both not-sua)
    changed = {by_label["good"][0]: "noise", by_label["noise"][0]: "good", by_label["MUA"][0]: "good"}
    unchanged = {by_label["MUA"][1]: "noise"}
    relabel(targets[0], {**changed, **unchanged})
    new_units = {"0": "good", "1": "noise", "2": "MUA"}
    relabel(targets[1], new_units)

    result = window.retrain_one_model(model_folder(project), f"{MODEL}_retrained_01", MODEL)

    assert result["num_units_added"] == len(changed) + len(new_units)
    assert result["num_corrected_labels"] == len(changed) and result["num_new_units"] == len(new_units)
    retrained = model_folder(project, f"{MODEL}_retrained_01")
    assert (retrained / "best_model.skops").is_file()

    # the units the model was trained on get your label in their own row; only the new units are new rows
    parent_labels = labels_by_row(model_folder(project))
    retrained_labels = labels_by_row(retrained)
    assert len(pd.read_csv(retrained / "training_data.csv")) == len(parent_labels) + len(new_units)
    assert len(retrained_labels) == len(parent_labels) + len(new_units)
    training = normalise_path(targets[0]["path"])
    expected = dict(parent_labels)
    expected.update({(training, unit_id): "not-sua" if label == "noise" else "good" for unit_id, label in changed.items()})
    new = normalise_path(targets[1]["path"])
    expected.update({(new, "0"): "good", (new, "1"): "not-sua", (new, "2"): "not-sua"})
    assert retrained_labels == expected

    # the retraining log, shown in the Retraining history
    log, units = read_log(retrained)
    assert log["parent"] == MODEL and log["num_units_added"] == 6
    assert log["num_same_as_original"] == 1
    assert 0 <= log["held_out"]["before"] <= 1 and 0 <= log["held_out"]["after"] <= 1
    added = dict(zip(zip(units["analyzer"], units["unit_id"]), units["taught_as"]))
    training, new = targets[0]["path"], targets[1]["path"]
    assert added == {
        **{(training, unit_id): "not-sua" if label == "noise" else "good" for unit_id, label in changed.items()},
        (new, "0"): "good", (new, "1"): "not-sua", (new, "2"): "not-sua",
    }
    assert dict(zip(units["unit_id"][units["analyzer"] == training], units["original_label"][units["analyzer"] == training])) == {
        unit_id: original[unit_id] for unit_id in changed}

    # the retrained model learned from the parent's units and the new ones
    trained_on, status = read_trained_on(retrained)
    assert status == "recorded"
    assert trained_on[normalise_path(targets[1]["path"])] == set(new_units)


def test_retraining_twice_does_not_add_new_units_again(window, project):
    targets = window.relabel_targets()
    relabel(targets[1], {"0": "good", "1": "noise"})
    window.retrain_one_model(model_folder(project), f"{MODEL}_retrained_01", MODEL)

    result = window.retrain_one_model(model_folder(project, f"{MODEL}_retrained_01"), f"{MODEL}_retrained_02", MODEL)
    assert result["retrained"] is False
    assert any("already added in an earlier retraining round" in reason for reason in result["reasons"])
    assert not model_folder(project, f"{MODEL}_retrained_02").exists()


def test_retraining_twice_does_not_add_relabelled_training_units_again(window, project):
    """A relabelled unit the model was trained on used to be added as a second row, with the other label, every round."""
    targets = window.relabel_targets()
    training_labels = pd.read_csv(Path(targets[0]["save_folder"]) / "labels.csv", dtype=str)
    good_unit = training_labels[training_labels["quality"] == "good"]["unit_id"].iloc[0]
    relabel(targets[0], {good_unit: "noise"})
    window.retrain_one_model(model_folder(project), f"{MODEL}_retrained_01", MODEL)

    result = window.retrain_one_model(model_folder(project, f"{MODEL}_retrained_01"), f"{MODEL}_retrained_02", MODEL)
    assert result["retrained"] is False
    assert result["reasons"] == ["1 units were already added in an earlier retraining round"]

    # relabelled again, with another label: the same row gets the new label
    relabel(targets[0], {good_unit: "good"}, key="another_selection")
    result = window.retrain_one_model(model_folder(project, f"{MODEL}_retrained_01"), f"{MODEL}_retrained_02", MODEL)
    assert result["num_corrected_labels"] == 1 and result["num_new_units"] == 0
    labels = labels_by_row(model_folder(project, f"{MODEL}_retrained_02"))
    assert labels == labels_by_row(model_folder(project))


def test_without_training_units_relabelled_units_are_added_as_new_rows(window, project):
    """Models trained before training_units.csv was saved, or downloaded ones, are retrained as before."""
    (model_folder(project) / "training_units.csv").unlink()
    targets = window.relabel_targets()
    training_labels = pd.read_csv(Path(targets[0]["save_folder"]) / "labels.csv", dtype=str)
    good_unit = training_labels[training_labels["quality"] == "good"]["unit_id"].iloc[0]
    relabel(targets[0], {good_unit: "noise"})
    result = window.retrain_one_model(model_folder(project), f"{MODEL}_retrained_01", MODEL)
    assert result["num_new_units"] == 1 and result["num_corrected_labels"] == 0
    retrained = model_folder(project, f"{MODEL}_retrained_01")
    assert len(pd.read_csv(retrained / "labels.csv")) == len(pd.read_csv(model_folder(project) / "labels.csv")) + 1
    assert not (retrained / "training_units.csv").exists()


def test_retraining_is_the_same_classifier_fitted_on_all_its_rows(window, project):
    """Retraining fits the parent's classifier (same settings) on the training split, the moved rows and the new rows."""
    import numpy as np
    from spikeinterface.curation.model_based_curation import load_model
    from unitrefine.predictions import as_model_input

    relabel(window.relabel_targets()[1], {"0": "good", "1": "noise", "2": "good"})
    window.retrain_one_model(model_folder(project), f"{MODEL}_retrained_01", MODEL)
    parent, _ = load_model(model_folder=model_folder(project), trust_model=True)
    retrained, _ = load_model(model_folder=model_folder(project, f"{MODEL}_retrained_01"), trust_model=True)
    assert retrained["classifier"].get_params() == parent["classifier"].get_params()
    # the preprocessing is the parent's, as fitted
    data = as_model_input(pd.read_csv(model_folder(project) / "training_data.csv").drop(columns="unit_id"))
    assert np.array_equal(parent[:-1].transform(data), retrained[:-1].transform(data))


def test_nothing_to_retrain_says_why(window, project):
    result = window.retrain_one_model(model_folder(project), f"{MODEL}_retrained_01", MODEL)
    assert result == dict(retrained=False, reasons=["you have not relabelled any units yet"], no_relabels=True)

    # relabels the model can't learn from: the same class as before
    targets = window.relabel_targets()
    training_labels = pd.read_csv(Path(targets[0]["save_folder"]) / "labels.csv", dtype=str)
    mua_unit = training_labels[training_labels["quality"] == "MUA"]["unit_id"].iloc[0]
    relabel(targets[0], {mua_unit: "noise"})
    result = window.retrain_one_model(model_folder(project), f"{MODEL}_retrained_01", MODEL)
    assert result["retrained"] is False and result["no_relabels"] is False
    assert result["reasons"] == ["1 units got the same label as before, in this model's classes"]


def test_sua_vs_mua_skips_noise_labels(window, project):
    relabel(window.relabel_targets()[1], {"0": "noise"})
    result = window.retrain_one_model(model_folder(project, "test_model_sua_vs_mua"), "retrained", "test_model_sua_vs_mua")
    assert result["retrained"] is False
    assert "1 units have a label this model doesn't predict" in result["reasons"][0]


def test_retrain_button_retrains_the_chosen_model(window, project, monkeypatch):
    import PyQt5.QtWidgets as QtWidgets

    summaries = []
    monkeypatch.setattr(QtWidgets.QMessageBox, "exec", lambda self: summaries.append(self.text()) or 0)
    selector = window.model_selector
    for row in range(selector.binary_list.count()):
        item = selector.binary_list.item(row)
        item.setCheckState(2 if item.data(256) == MODEL else 0)
    relabel(window.relabel_targets()[1], {"0": "good", "1": "noise"})

    window.retrainedModelNameForm.setText(f"{MODEL}_retrained_01")
    window.retrain_model()

    assert (model_folder(project, f"{MODEL}_retrained_01") / "best_model.skops").is_file()
    assert "learned from 2" in summaries[-1]
    # the retrained model is added to the project
    assert any(Path(path).name == f"{MODEL}_retrained_01" for path, _ in window.project.model_paths)


def test_relabelled_held_out_units_are_learned_not_held_out(window, project):
    """Units you relabel are learned even if they were in the held-out split; the others stay held out."""
    import numpy as np
    from sklearn.model_selection import train_test_split

    codes = pd.read_csv(model_folder(project) / "labels.csv").drop(columns="unit_index").iloc[:, 0].to_numpy()
    _, held_out = train_test_split(np.arange(len(codes)), test_size=0.2, random_state=0, stratify=codes)
    training_units = pd.read_csv(model_folder(project) / "training_units.csv", dtype=str)
    flip = {"good": "noise", "MUA": "good", "noise": "good"}
    original = dict(zip(*[pd.read_csv(Path(window.relabel_targets()[0]["save_folder"]) / "labels.csv", dtype=str)[column]
                          for column in ["unit_id", "quality"]]))
    # two of the held-out units, which change class
    relabelled = [training_units["unit_id"][row] for row in held_out[:2]]
    relabel(window.relabel_targets()[0], {unit_id: flip[original[unit_id]] for unit_id in relabelled})

    result = window.retrain_one_model(model_folder(project), f"{MODEL}_retrained_01", MODEL)
    assert result["num_corrected_labels"] == 2
    assert result["num_held_out"] == len(held_out) - 2
