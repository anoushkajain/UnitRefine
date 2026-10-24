"""The figures of sections 3 and 4: every figure is made without an error, and saved figures are reused."""

import os
import shutil
import stat
from pathlib import Path

import pandas as pd
import pytest

from conftest import targets_of
from unitrefine import figures
from unitrefine.main import MainWindow, load_project
from unitrefine.predictions import find_least_confident_units, relabelled_units_path, units_to_relabel_path

MODEL = "test_model_sua_vs_all"
CASCADE = ["test_model_noise_vs_neural", "test_model_sua_vs_mua"]


@pytest.fixture(scope="module")
def project(tmp_path_factory, trained_project, recordings, qapp):
    """
    The trained project with the new recording (without labels), after one round of active learning with
    the SUA vs all model: units found, relabelled and the model retrained.
    """
    folder = tmp_path_factory.mktemp("figures") / "project"
    shutil.copytree(trained_project, folder)
    loaded = load_project(folder)
    loaded.add_analyzer(str(recordings["new"]))
    loaded.save()

    models = folder / "models"
    targets = targets_of(folder)
    find_least_confident_units(targets, "binary", [models / MODEL], MODEL, 6)
    picked = pd.read_csv(units_to_relabel_path(targets[1], MODEL), dtype=str)["unit_id"]
    relabels = pd.DataFrame({"quality": [["good", "MUA", "noise"][index % 3] for index in range(len(picked))], "unit_id": picked})
    relabels.to_csv(relabelled_units_path(targets[1], MODEL), index=False)

    window = MainWindow(load_project(folder))
    window.retrain_one_model(models / MODEL, f"{MODEL}_retrained_01", MODEL)
    window.close()
    return folder


def all_specs(project):
    targets = targets_of(project)
    models = project / "models"
    retrained = models / f"{MODEL}_retrained_01"
    cascade = [models / name for name in CASCADE]
    cascade_key = "cascade__" + "__".join(CASCADE)
    return (
        figures.training_specs(project, targets, [models / MODEL, *cascade])
        + figures.apply_specs(project, targets, "binary", [models / MODEL], MODEL)
        + figures.apply_specs(project, targets, "binary", [models / MODEL, cascade[0]], f"{MODEL}+{CASCADE[0]}")
        + figures.apply_specs(project, targets, "cascade", cascade, cascade_key)
        + figures.active_learning_specs(project, targets, "binary", [retrained], retrained.name,
                                        {MODEL: [models / MODEL, retrained]})
        + figures.active_learning_specs(project, targets, "cascade", cascade, cascade_key,
                                        {name: [models / name] for name in CASCADE})
    )


def test_every_figure_is_made_without_an_error(project):
    specs = all_specs(project)
    results = [figures.render(spec) for spec in specs]

    errors = [(spec.path.name, result["message"]) for spec, result in zip(specs, results) if result["status"] == "error"]
    assert errors == []
    for spec, result in zip(specs, results):
        if result["status"] == "done":
            assert spec.path.is_file()
        else:
            # a figure which can't be made yet says what to do first
            assert result["status"] == "nodata" and result["message"]
    not_made = [spec.key for spec, result in zip(specs, results) if result["status"] != "done"]
    # only the cascade, which wasn't retrained, has no relabels and rounds to show
    assert not_made == ["relabel_changes", "rounds"]


def test_figures_made_by_two_threads_at_once_are_made_one_at_a_time(project):
    """
    Each window makes its figures in a thread, and a figure starts by resetting the record of the analyzers it uses:
    two figures made at the same time (e.g. by a window's thread and a test) must not reset each other's.
    """
    import threading

    specs = [spec for spec in all_specs(project) if spec.key in ("predicted_counts", "confidence")]
    results = []

    def make_figures():
        for _ in range(4):
            results.extend(figures.render(spec) for spec in specs)

    threads = [threading.Thread(target=make_figures) for _ in range(3)]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join()
    assert [result["message"] for result in results if result["status"] == "error"] == []


def test_figures_are_reused_until_their_data_changes(project):
    spec = [spec for spec in figures.training_specs(project, targets_of(project), [project / "models" / MODEL])
            if "confusion" in spec.key.lower() or "test" in spec.key.lower()][0]
    figures.render(spec)
    assert figures.cached_result(spec) is not None

    # a changed input file makes the figure out of date
    input_path = Path(spec.inputs()[0])
    original = input_path.read_bytes()
    input_path.write_bytes(original + b"\n")
    try:
        assert figures.cached_result(spec) is None
    finally:
        input_path.write_bytes(original)


def test_a_figure_which_cant_be_saved_is_reported(project):
    spec = figures.training_specs(project, targets_of(project), [project / "models" / MODEL])[0]
    figures.render(spec)
    os.chmod(spec.path, stat.S_IREAD)
    try:
        # the figure is made again, as its record is removed
        figures.sidecar_path(spec).unlink()
        result = figures.render(spec)
    finally:
        os.chmod(spec.path, stat.S_IWRITE | stat.S_IREAD)
    if result["status"] == "done":
        pytest.skip("this system lets the owner replace read-only files")
    assert result["status"] == "error" and "Read-only" in result["message"]
