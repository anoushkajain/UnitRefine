"""Apply in section 3.3 offers to compute the metrics the models use which an analyzer doesn't have."""

import time

import pandas as pd
import pytest

from test_cli import SUA_VS_ALL, analyzer_without_metrics


@pytest.fixture
def window(trained_project, project_copy, qapp, monkeypatch):
    """The main window of a copy of the trained project, with the SUA vs all model chosen, and its messages."""
    import PyQt5.QtWidgets as QtWidgets
    from unitrefine.main import MainWindow, load_project

    window = MainWindow(load_project(project_copy(trained_project)))
    selector = window.model_selector
    for row in range(selector.binary_list.count()):
        item = selector.binary_list.item(row)
        item.setCheckState(2 if item.data(256) == SUA_VS_ALL else 0)
    window.messages = []
    for name in ["information", "warning"]:
        monkeypatch.setattr(QtWidgets.QMessageBox, name,
                            staticmethod(lambda parent, title, text, *args: window.messages.append((title, text))))
    monkeypatch.setattr(type(window), "metrics_options", ["--n_jobs", "1"])
    yield window
    window.close()


def answer_confirm(window, monkeypatch, answer):
    questions = []
    monkeypatch.setattr(window, "confirm", lambda title, text: questions.append((title, text)) or answer)
    return questions


def wait_for_metrics(window, qapp):
    end = time.time() + 600
    while window.metrics_process is not None and time.time() < end:
        qapp.processEvents()
        time.sleep(0.05)
    assert window.metrics_process is None, "the metrics were not computed in 10 minutes"


def applied_paths(window):
    return [entry["path"] for entry in window.project.config.get("applied_analyzers", [])]


def test_the_missing_metrics_are_computed_and_the_models_applied(window, tmp_path, qapp, monkeypatch):
    analyzer_folder = analyzer_without_metrics(tmp_path / "no_metrics")
    questions = answer_confirm(window, monkeypatch, True)
    added = []

    window.apply_to_new_analyzers([str(analyzer_folder)], after_added=lambda: added.append(True))
    assert window.metrics_process is not None
    # the window can't be used while the metrics are computed
    assert not window.content_widget.isEnabled()
    wait_for_metrics(window, qapp)

    assert questions[0][0] == "Compute the missing metrics?" and "snr" in questions[0][1]
    assert window.messages == []
    assert applied_paths(window) == [str(analyzer_folder)]
    applied = window.project.config["applied_analyzers"][0]
    predictions = pd.read_csv(f"{applied['output_folder']}/{applied['output_name']}__labels_from_{SUA_VS_ALL}.csv")
    assert len(predictions) == 6
    assert added == [True] and window.content_widget.isEnabled()


def test_without_computing_them_the_analyzer_is_not_added(window, tmp_path, monkeypatch):
    analyzer_folder = analyzer_without_metrics(tmp_path / "no_metrics")
    answer_confirm(window, monkeypatch, False)

    window.apply_to_new_analyzers([str(analyzer_folder)])

    assert window.metrics_process is None and applied_paths(window) == []
    assert window.messages[0][0] == "Some analyzers were not added"
    assert "required metrics" in window.messages[0][1]


def test_metrics_which_cant_be_computed_are_reported(window, tmp_path, qapp, monkeypatch):
    analyzer_folder = analyzer_without_metrics(tmp_path / "no_recording", with_recording=False)
    answer_confirm(window, monkeypatch, True)

    window.apply_to_new_analyzers([str(analyzer_folder)])
    wait_for_metrics(window, qapp)

    assert applied_paths(window) == []
    assert window.messages[0][0] == "Could not compute the metrics"
    assert "The analyzer has no recording" in window.messages[0][1]
