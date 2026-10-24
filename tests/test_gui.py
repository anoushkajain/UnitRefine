"""The UnitRefine window: it opens, every button works, and sections fold. Dialogs are answered as if cancelled."""

import sys
import time
import traceback

import PyQt5.QtWidgets as QtWidgets
import pytest
from PyQt5.QtCore import QProcess

import unitrefine.main as main
import unitrefine.widgets as widgets
from unitrefine.main import MainWindow, UnitRefineProject, load_project
from unitrefine.widgets import CollapsibleSection, FigureCard


@pytest.fixture
def errors(monkeypatch):
    """Errors raised in the window's slots; without an excepthook, PyQt would close the test session."""
    found = []
    monkeypatch.setattr(sys, "excepthook", lambda kind, value, trace: found.append("".join(traceback.format_exception(kind, value, trace))))
    return found


@pytest.fixture
def no_dialogs(monkeypatch):
    """Dialogs return at once, as if cancelled; the SpikeInterface-GUI and links are recorded, not opened."""
    launched = []
    monkeypatch.setattr(QProcess, "start", lambda self, program, arguments=(): launched.append(list(arguments)))
    monkeypatch.setattr(QtWidgets.QFileDialog, "getOpenFileName", staticmethod(lambda *args, **kwargs: ("", "")))
    monkeypatch.setattr(QtWidgets.QFileDialog, "getSaveFileName", staticmethod(lambda *args, **kwargs: ("", "")))
    monkeypatch.setattr(QtWidgets.QFileDialog, "getExistingDirectory", staticmethod(lambda *args, **kwargs: ""))
    monkeypatch.setattr(QtWidgets.QFileDialog, "exec", lambda self: False)
    monkeypatch.setattr(QtWidgets.QDialog, "exec", lambda self: False)
    monkeypatch.setattr(QtWidgets.QMessageBox, "exec", lambda self: 0)
    for name in ["information", "warning", "critical"]:
        monkeypatch.setattr(QtWidgets.QMessageBox, name, staticmethod(lambda *args, **kwargs: 0))
    monkeypatch.setattr(QtWidgets.QMessageBox, "question", staticmethod(lambda *args, **kwargs: QtWidgets.QMessageBox.No))
    for module in [main, widgets]:
        monkeypatch.setattr(module.QDesktopServices, "openUrl", staticmethod(lambda url: launched.append(["open", url.toString()])))
    return launched


def pump(app, seconds):
    end = time.time() + seconds
    while time.time() < end:
        app.processEvents()
        time.sleep(0.02)


def close(window, app):
    window.close()
    pump(app, 0.2)


def test_a_new_project_opens_with_only_section_3(tmp_path, qapp, errors):
    project = UnitRefineProject(tmp_path / "new_project")
    project.save()
    window = MainWindow(project)
    window.finish_loading()
    expanded = {section.number: section.expanded for section in window.findChildren(CollapsibleSection)}
    assert expanded == {1: False, 2: False, 3: True, 4: False, 5: False}
    close(window, qapp)
    assert errors == []


def test_folded_sections_are_remembered(project_copy, labelled_project, qapp, errors):
    folder = project_copy(labelled_project)
    window = MainWindow(load_project(folder))
    sections = {section.number: section for section in window.findChildren(CollapsibleSection)}
    sections[4].set_expanded(True)
    sections[3].set_expanded(False)
    close(window, qapp)

    window = MainWindow(load_project(folder))
    expanded = {section.number: section.expanded for section in window.findChildren(CollapsibleSection)}
    assert expanded[4] is True and expanded[3] is False
    close(window, qapp)
    assert errors == []


def test_every_button_works(project_copy, trained_project, recordings, qapp, errors, no_dialogs):
    folder = project_copy(trained_project)
    project = load_project(folder)
    project.add_analyzer(str(recordings["new"]))
    project.save()
    window = MainWindow(load_project(folder))
    window.finish_loading()
    window.show()
    for section in window.findChildren(CollapsibleSection):
        section.set_expanded(True)
    pump(qapp, 0.2)

    clicked = []
    for button in window.findChildren(QtWidgets.QPushButton) + window.findChildren(QtWidgets.QToolButton):
        # removing an analyzer or model is tested in test_project.py
        if not button.isVisibleTo(window) or button.toolTip().startswith("Remove"):
            continue
        name = button.text() or button.toolTip()
        num_errors = len(errors)
        button.click()
        pump(qapp, 0.05)
        assert len(errors) == num_errors, f"clicking '{name}':\n{errors[-1]}"
        clicked.append(name)
        # the SpikeInterface-GUI never opens here, so carry on as if it had been closed
        if window.sigui_process is not None:
            window.sigui_process = None
            window.sigui_opening_dialog.done_opening()
            window.content_widget.setEnabled(True)
    assert len(clicked) > 15
    # Curate, Inspect and Relabel launch the SpikeInterface-GUI
    assert any(any("launch_sigui" in str(argument) for argument in arguments) for arguments in no_dialogs)

    # the figures are made in the background; a figure card opens the viewer
    pump(qapp, 5)
    cards = window.findChildren(FigureCard)
    assert len(cards) > 0
    for card in cards[:3]:
        card.clicked.emit()
    pump(qapp, 0.2)
    close(window, qapp)
    assert errors == []
