"""
Labels set in the SpikeInterface-GUI are saved when it closes, in Curate, Inspect and Relabel. Each test runs a
launch script in its own process, as UnitRefine does, with sigui_harness.py setting labels as the keys do.
"""

import os
import subprocess
import sys
from pathlib import Path

import pandas as pd
import pytest

import unitrefine
from unitrefine.predictions import predictions_file_name, relabelled_units_path

pytest.importorskip("spikeinterface_gui")

SCRIPTS = Path(unitrefine.__file__).parent
HARNESS = Path(__file__).parent / "sigui_harness.py"
MODEL = "test_model_sua_vs_all"


def run_sigui(script, arguments, labels):
    command = [sys.executable, str(HARNESS), str(SCRIPTS / script), *map(str, arguments),
               "--labels", ",".join(f"{unit_id}={label}" for unit_id, label in labels.items())]
    environment = dict(os.environ, QT_QPA_PLATFORM="offscreen")
    finished = subprocess.run(command, capture_output=True, text=True, env=environment, timeout=600)
    assert finished.returncode == 0, finished.stdout[-3000:] + finished.stderr[-3000:]
    shown = [line for line in finished.stdout.splitlines() if line.startswith("SHOWN_UNITS")]
    assert shown, finished.stdout[-3000:]
    return shown[0].split(" ", 1)[1].split(",") if " " in shown[0] else []


def read(path):
    table = pd.read_csv(path, dtype=str)
    return dict(zip(table["unit_id"], table["quality"]))


def test_curate_saves_the_labels(project_copy, labelled_project, recordings):
    folder = project_copy(labelled_project)
    labels_path = folder / "analyzers" / "analyzer_0" / "labels.csv"
    before = read(labels_path)
    changed = {"0": "noise" if before["0"] != "noise" else "good", "1": "MUA" if before["1"] != "MUA" else "good"}

    run_sigui("launch_sigui.py", [recordings["training"], folder, 0], changed)

    # the changed labels are saved, and the other labels kept: they were shown in the quality column
    assert read(labels_path) == {**before, **changed}
    assert len(pd.read_csv(folder / "analyzers" / "analyzer_0" / "labelled_metrics.csv")) == 30
    assert len(pd.read_csv(folder / "analyzers" / "analyzer_0" / "all_metrics.csv")) == 30


def test_curate_a_new_analyzer(project_copy, labelled_project, recordings):
    from unitrefine.main import load_project

    folder = project_copy(labelled_project)
    project = load_project(folder)
    project.add_analyzer(str(recordings["new"]))
    project.save()

    run_sigui("launch_sigui.py", [recordings["new"], folder, 1], {"3": "good", "4": "noise"})
    assert read(folder / "analyzers" / "analyzer_1" / "labels.csv") == {"3": "good", "4": "noise"}


def inspect_arguments(folder, recordings, *extra):
    save_folder = folder / "analyzers" / "analyzer_0"
    return [recordings["training"], "--save_folder", save_folder, "--mode", "binary",
            "--model_folders", folder / "models" / MODEL, "--selection_key", MODEL,
            "--human_labels", save_folder / "labels.csv", *extra]


def target(folder):
    return dict(save_folder=folder / "analyzers" / "analyzer_0", file_prefix="")


def test_inspect_saves_only_labels_which_changed(project_copy, trained_project, recordings):
    folder = project_copy(trained_project)
    original = read(folder / "analyzers" / "analyzer_0" / "labels.csv")
    new_label = "noise" if original["0"] != "noise" else "good"
    # unit 1 gets the label it already has
    shown = run_sigui("launch_sigui_validate.py", inspect_arguments(folder, recordings), {"0": new_label, "1": original["1"]})

    assert len(shown) == 30
    assert read(relabelled_units_path(target(folder), MODEL)) == {"0": new_label}
    # the human labels are not changed by Inspect
    assert read(folder / "analyzers" / "analyzer_0" / "labels.csv") == original
    # the predictions are saved, for the figures and the next Inspect
    predictions = pd.read_csv(folder / "analyzers" / "analyzer_0" / predictions_file_name("", MODEL))
    assert len(predictions) == 30 and f"{MODEL}_model_label" in predictions.columns


def test_relabel_shows_only_the_units_to_relabel_and_keeps_earlier_labels(project_copy, trained_project, recordings, tmp_path):
    folder = project_copy(trained_project)
    original = read(folder / "analyzers" / "analyzer_0" / "labels.csv")
    relabels_path = relabelled_units_path(target(folder), MODEL)
    # a label from an earlier round, for a unit which is not shown this time
    pd.DataFrame({"quality": ["MUA"], "unit_id": ["20"]}).to_csv(relabels_path, index=False)
    units_file = tmp_path / "units_to_relabel.csv"
    pd.DataFrame({"unit_id": ["5", "6", "7"], "confidence": [0.5, 0.55, 0.6]}).to_csv(units_file, index=False)

    new_labels = {"5": "noise" if original["5"] != "noise" else "good", "6": "MUA" if original["6"] != "MUA" else "good"}
    shown = run_sigui("launch_sigui_validate.py",
                      inspect_arguments(folder, recordings, "--relabel", "--units_file", units_file), new_labels)

    assert shown == ["5", "6", "7"]
    # unit 7 was shown without a label and not labelled
    assert read(relabels_path) == {"20": "MUA", **new_labels}
