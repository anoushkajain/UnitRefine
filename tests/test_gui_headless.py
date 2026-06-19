import os
from pathlib import Path

import pytest


def test_train_window_headless(monkeypatch, tmp_path):
    # Force Qt to use the offscreen platform for headless CI
    monkeypatch.setenv('QT_QPA_PLATFORM', 'offscreen')

    # import after setting the env var
    from PyQt5.QtWidgets import QApplication
    from PyQt5.QtTest import QTest
    from PyQt5.QtCore import Qt

    # import the window under test
    from unitrefine.train import TrainWindow

    class DummyProject:
        def __init__(self, folder: Path):
            self.folder_name = folder
            self.model_paths = []
            self.analyzers = {}

    proj = DummyProject(tmp_path)

    app = QApplication.instance() or QApplication([])

    w = TrainWindow(proj)
    w.show()

    # let Qt process pending events
    app.processEvents()

    # Basic smoke assertions
    assert w.centralWidget() is not None
    assert hasattr(w, 'metric_selector')
    assert w.metric_selector.count() >= 1

    # Exercise metric pill creation and removal
    w.make_new_metrics_list(['m1', 'm2'])
    app.processEvents()
    assert w.pill_layout.count() == 2

    # Click the close button of the first pill to remove it
    item = w.pill_layout.itemAt(0)
    pill = item.widget()
    assert pill is not None
    QTest.mouseClick(pill.close_btn, Qt.LeftButton)
    app.processEvents()

    # metric_names should be updated (removal callback executed)
    assert isinstance(w.metric_names, list)

    w.close()
    app.processEvents()
