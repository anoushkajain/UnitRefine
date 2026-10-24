"""
Shared test data: small synthetic recordings (made with SpikeInterface, no data files needed), a UnitRefine
project with imported labels, and models trained on it. The slow parts (recordings, training) are made once
per test session; tests which change a project work on their own copy.
"""

import os
import shutil
import sys
from pathlib import Path

# the GUI tests run without a screen
os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
if sys.platform == "win32":
    os.environ.setdefault("QT_QPA_FONTDIR", "C:/Windows/Fonts")

import numpy as np
import pandas as pd
import pytest

LABELS = ["good", "MUA", "noise"]


def make_analyzer(folder, num_units, seed, integer_ids=False):
    """A synthetic recording with the extensions and metrics UnitRefine and the SpikeInterface-GUI use."""
    import spikeinterface.full as si

    recording, sorting = si.generate_ground_truth_recording(durations=[60.0], num_units=num_units, seed=seed)
    if integer_ids:
        sorting = sorting.rename_units(list(range(num_units)))
    analyzer = si.create_sorting_analyzer(sorting, recording, folder=folder, format="binary_folder", sparse=True)
    analyzer.compute(["random_spikes", "templates", "noise_levels", "unit_locations", "quality_metrics", "template_metrics"])
    return folder


def write_labels(path, unit_ids, column="quality", id_column="unit_id", sep=","):
    """Labels good, MUA, noise in turn, so every class has units."""
    labels = pd.DataFrame({id_column: list(unit_ids), column: [LABELS[index % 3] for index in range(len(unit_ids))]})
    labels.to_csv(path, sep=sep, index=False)
    return dict(zip([str(unit_id) for unit_id in unit_ids], labels[column]))


@pytest.fixture(scope="session")
def qapp():
    import PyQt5.QtWidgets as QtWidgets
    from unitrefine import theme

    app = QtWidgets.QApplication.instance() or QtWidgets.QApplication([])
    app.setStyleSheet(theme.STYLE_SHEET)
    return app


@pytest.fixture(scope="session")
def recordings(tmp_path_factory):
    """Two recordings: one to train on (30 units) and a new one (25 units), both with text unit ids."""
    folder = tmp_path_factory.mktemp("recordings")
    return dict(
        training=make_analyzer(folder / "training_analyzer", num_units=30, seed=0),
        new=make_analyzer(folder / "new_analyzer", num_units=25, seed=1),
    )


@pytest.fixture(scope="session")
def labelled_project(tmp_path_factory, recordings):
    """A project with the training recording and labels imported for all its units."""
    import spikeinterface.full as si
    from unitrefine.import_labels import import_labels
    from unitrefine.main import UnitRefineProject

    folder = tmp_path_factory.mktemp("projects") / "labelled_project"
    project = UnitRefineProject(folder)
    project.save()
    project.add_analyzer(str(recordings["training"]))
    project.save()
    unit_ids = si.load_sorting_analyzer(recordings["training"], load_extensions=False).unit_ids
    label_file = folder.parent / "labels.csv"
    write_labels(label_file, unit_ids)
    import_labels(label_file, str(recordings["training"]), folder / "analyzers" / "analyzer_0")
    return folder


@pytest.fixture(scope="session")
def trained_project(tmp_path_factory, labelled_project, qapp):
    """The labelled project with three models trained in the Train window: noise vs neural, SUA vs MUA, SUA vs all."""
    import PyQt5.QtWidgets as QtWidgets
    from unitrefine.main import load_project
    from unitrefine.train import TrainWindow

    folder = tmp_path_factory.mktemp("trained") / "trained_project"
    shutil.copytree(labelled_project, folder)
    window = TrainWindow(load_project(folder))
    window.modelNameForm.setText("test_model")
    for checkbox in window.task_checkboxes.values():
        checkbox.setChecked(True)
    train_button = [button for button in window.findChildren(QtWidgets.QPushButton) if button.text() == "Train models"][0]
    train_button.click()
    window.close()
    for task in ["noise_vs_neural", "sua_vs_mua", "sua_vs_all"]:
        assert (folder / "models" / f"test_model_{task}" / "best_model.skops").is_file(), f"{task} was not trained"
    return folder


@pytest.fixture
def project_copy(tmp_path):
    """Copies a session project, so a test can change it."""
    def copy(project_folder):
        destination = tmp_path / Path(project_folder).name
        shutil.copytree(project_folder, destination)
        return destination
    return copy


def targets_of(project_folder, applied=()):
    """The relabel targets (see MainWindow.relabel_targets) of a project, without opening the window."""
    from unitrefine.main import load_project

    project = load_project(project_folder)
    targets = []
    for index, analyzer in enumerate(project.analyzers.values()):
        targets.append(dict(path=str(analyzer["path"]), save_folder=Path(project_folder) / analyzer["analyzer_in_project"],
                            file_prefix="", analyzer_index=index, analyzer=analyzer))
    for path, output_folder, output_name in applied:
        targets.append(dict(path=str(path), save_folder=Path(output_folder), file_prefix=f"{output_name}__",
                            analyzer_index=None, analyzer={"path": str(path)}))
    return targets
