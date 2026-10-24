"""The project folder: saving, adding and removing analyzers, and files which can't be written."""

import os
import stat

import pandas as pd
import pytest

from conftest import make_analyzer
from unitrefine.main import MainWindow, UnitRefineProject, load_project


def labels_of(project_folder, index=0):
    labels = pd.read_csv(project_folder / "analyzers" / f"analyzer_{index}" / "labels.csv")
    return dict(zip(labels["unit_id"].astype(str), labels["quality"]))


def test_saving_the_project_keeps_labels_set_after_opening_it(project_copy, labelled_project, recordings):
    """Adding an analyzer saves the project; this used to overwrite labels.csv with the copy read when opening it."""
    folder = project_copy(labelled_project)
    project = load_project(folder)
    # labels change after the project was opened, e.g. in Curate (which writes labels.csv itself)
    labels_path = folder / "analyzers" / "analyzer_0" / "labels.csv"
    pd.DataFrame({"quality": ["noise"], "unit_id": ["0"]}).to_csv(labels_path, index=False)

    project.add_analyzer(str(recordings["new"]))
    project.save()

    assert labels_of(folder) == {"0": "noise"}
    assert (folder / "analyzers" / "analyzer_1" / "path.txt").read_text() == str(recordings["new"])


def test_load_project_skips_folders_without_an_analyzer(project_copy, labelled_project):
    folder = project_copy(labelled_project)
    # e.g. left over from a removed analyzer
    (folder / "analyzers" / "analyzer_7").mkdir()
    project = load_project(folder)
    assert list(project.analyzers) == [0]


def test_a_new_analyzer_does_not_reuse_a_leftover_folder(project_copy, labelled_project, recordings):
    folder = project_copy(labelled_project)
    (folder / "analyzers" / "analyzer_7").mkdir()
    project = load_project(folder)
    project.add_analyzer(str(recordings["new"]))
    assert project.analyzers[8]["analyzer_in_project"].name == "analyzer_8"


def test_analyzers_are_loaded_in_the_order_they_were_added(tmp_path, recordings):
    """By number, whatever order the folders are listed in (any order on Linux, alphabetical on Windows)."""
    project = UnitRefineProject(tmp_path / "project")
    project.save()
    for _ in range(12):
        project.add_analyzer(str(recordings["new"]))
    project.save()
    # analyzer 1 is written last, so it is not listed first anywhere
    (tmp_path / "project" / "analyzers" / "analyzer_1" / "path.txt").write_text(str(recordings["training"]))
    loaded = load_project(tmp_path / "project")
    assert list(loaded.analyzers) == list(range(12))
    assert [analyzer["analyzer_in_project"] for analyzer in loaded.analyzers.values()] == [f"analyzers/analyzer_{index}" for index in range(12)]


def test_removing_an_analyzer_asks_first(project_copy, labelled_project, qapp, monkeypatch):
    folder = project_copy(labelled_project)
    window = MainWindow(load_project(folder))
    questions = []

    monkeypatch.setattr(window, "confirm", lambda title, text: questions.append(text) or False)
    window.remove_analyzer(0)
    assert "30 labels" in questions[0]
    assert (folder / "analyzers" / "analyzer_0" / "labels.csv").is_file()
    assert 0 in window.project.analyzers

    monkeypatch.setattr(window, "confirm", lambda title, text: True)
    window.remove_analyzer(0)
    assert not (folder / "analyzers" / "analyzer_0").exists()
    assert 0 not in window.project.analyzers
    window.close()


def test_a_read_only_config_is_reported_not_raised(tmp_path, qapp, monkeypatch):
    project = UnitRefineProject(tmp_path / "project")
    project.save()
    config_path = tmp_path / "project" / "config.json"
    os.chmod(config_path, stat.S_IREAD)
    reported = []
    monkeypatch.setattr(UnitRefineProject, "on_unwritable_file", staticmethod(lambda path, message: reported.append(message)))
    try:
        project.config["setting"] = 1
        project.save_config()
    finally:
        os.chmod(config_path, stat.S_IWRITE | stat.S_IREAD)
    if os.access(config_path, os.W_OK) and not reported:
        pytest.skip("this system lets the owner write read-only files (e.g. running as root)")
    assert "config.json" in reported[0] and "Read-only" in reported[0]
