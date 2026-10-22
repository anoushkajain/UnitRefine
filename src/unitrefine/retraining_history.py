"""
What each retraining round did: which units were added from which recording, their labels before and after,
and the held-out balanced accuracy before and after. Saved with each retrained model
(retraining_log.json, and retraining_units.csv for the units), and shown in section 4.
"""

import json
from datetime import datetime
from pathlib import Path

import pandas as pd
import PyQt5.QtWidgets as QtWidgets
from PyQt5.QtCore import Qt

from unitrefine import theme

LOG_FILE = "retraining_log.json"
UNITS_FILE = "retraining_units.csv"

UNIT_COLUMNS = ["analyzer", "group", "unit_id", "original_label", "model_label", "your_label", "taught_as"]
UNIT_COLUMN_TITLES = {
    "analyzer": "Recording",
    "group": "Training or new",
    "unit_id": "Unit id",
    "original_label": "Original label",
    "model_label": "Model's label",
    "your_label": "Your label",
    "taught_as": "Taught as",
}


def write_log(retrained_model_folder, log, units):
    """Saves the round's summary (`log`) and its units (a list of dicts with UNIT_COLUMNS)."""
    retrained_model_folder = Path(retrained_model_folder)
    log = dict(log, date=datetime.now().isoformat(timespec="minutes"), num_units_added=len(units))
    with open(retrained_model_folder / LOG_FILE, "w") as f:
        json.dump(log, f, indent=4)
    pd.DataFrame(units, columns=UNIT_COLUMNS).to_csv(retrained_model_folder / UNITS_FILE, index=False)


def read_log(model_folder):
    """The round's summary and units (a DataFrame), or (None, None) for models retrained before this was recorded."""
    model_folder = Path(model_folder)
    if not (model_folder / LOG_FILE).is_file():
        return None, None
    with open(model_folder / LOG_FILE) as f:
        log = json.load(f)
    units_path = model_folder / UNITS_FILE
    units = pd.read_csv(units_path, dtype=str).fillna("") if units_path.is_file() else pd.DataFrame(columns=UNIT_COLUMNS)
    return log, units


def short_path(path):
    parts = Path(str(path).rstrip("/\\")).parts
    return "/".join(parts[-2:])


def accuracy_text(log):
    held_out = log.get("held_out") or {}
    before, after = held_out.get("before"), held_out.get("after")
    if before is None or after is None:
        return "—", None
    change = after - before
    return f"{before:.2f} → {after:.2f}  ({change:+.2f})", change


def recordings_text(units):
    if units is None or len(units) == 0:
        return "—"
    counts = units.groupby("analyzer").size()
    return ", ".join(f"{short_path(path)} ({count})" for path, count in counts.items())


class RetrainingHistoryTable(QtWidgets.QTableWidget):
    """One row per retraining round of the chosen models: what was added, from where, and the change in accuracy."""

    COLUMNS = ["Round", "Model", "Date", "Units added", "Recordings (units)", "Held-out balanced accuracy", ""]

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setColumnCount(len(self.COLUMNS))
        self.setHorizontalHeaderLabels(self.COLUMNS)
        self.setEditTriggers(QtWidgets.QAbstractItemView.NoEditTriggers)
        self.setSelectionMode(QtWidgets.QAbstractItemView.NoSelection)
        self.setFocusPolicy(Qt.NoFocus)
        self.verticalHeader().setVisible(False)
        self.setShowGrid(False)
        self.setWordWrap(False)
        self.setVerticalScrollBarPolicy(Qt.ScrollBarAlwaysOff)
        header = self.horizontalHeader()
        header.setStretchLastSection(False)
        for column in range(len(self.COLUMNS)):
            header.setSectionResizeMode(column, QtWidgets.QHeaderView.ResizeToContents)
        header.setSectionResizeMode(4, QtWidgets.QHeaderView.Stretch)
        header.setDefaultAlignment(Qt.AlignLeft | Qt.AlignVCenter)

    def set_lineages(self, lineages):
        """`lineages`: {base name: [original model folder, retrained_01 folder, ...]}."""
        rows = []
        for base_name, folders in lineages.items():
            for round_number, folder in enumerate(folders[1:], start=1):
                rows.append((round_number, Path(folder), folders[round_number - 1]))
        self.setRowCount(len(rows))
        for row, (round_number, folder, parent_folder) in enumerate(rows):
            log, units = read_log(folder)
            if log is None:
                values = [str(round_number), folder.name, "", "", "not recorded (retrained with an older version)", "—"]
                change = None
            else:
                accuracy, change = accuracy_text(log)
                values = [str(round_number), folder.name, log.get("date", "").replace("T", " "),
                          str(log.get("num_units_added", len(units))), recordings_text(units), accuracy]
            for column, value in enumerate(values):
                item = QtWidgets.QTableWidgetItem(value)
                item.setToolTip(value)
                if column == 5 and change is not None:
                    item.setForeground(Qt.darkGreen if change > 0 else (Qt.darkRed if change < 0 else Qt.black))
                self.setItem(row, column, item)
            details_button = theme.make_flat(QtWidgets.QPushButton("Details"))
            details_button.setEnabled(log is not None)
            details_button.clicked.connect(lambda _checked=False, folder=folder, round_number=round_number: self.show_details(folder, round_number))
            self.setCellWidget(row, 6, details_button)
        self.resizeRowsToContents()
        height = self.horizontalHeader().height() + sum(self.rowHeight(row) for row in range(self.rowCount())) + 4
        self.setFixedHeight(height)
        return len(rows)

    def show_details(self, folder, round_number):
        RetrainingDetailsDialog(folder, round_number, parent=self.window()).show()


class RetrainingDetailsDialog(QtWidgets.QDialog):
    """Every unit a retraining round added: recording, unit id, original label, the model's label and yours."""

    def __init__(self, folder, round_number, parent=None):
        super().__init__(parent)
        self.setAttribute(Qt.WidgetAttribute.WA_DeleteOnClose)
        log, units = read_log(folder)
        self.setWindowTitle(f"UnitRefine: retraining round {round_number}")

        layout = QtWidgets.QVBoxLayout(self)
        layout.setContentsMargins(20, 16, 20, 16)
        layout.setSpacing(10)

        title = QtWidgets.QLabel(f"Round {round_number}: {log.get('parent', '?')} → {Path(folder).name}")
        title.setStyleSheet("font-weight: bold; font-size: 13pt;")
        layout.addWidget(title)

        accuracy, _ = accuracy_text(log)
        held_out = log.get("held_out") or {}
        summary = (
            f"<b>{log.get('num_units_added', len(units))} units added</b> to the "
            f"{log.get('num_training_units_before', '?')} training units of {log.get('parent', 'the model')}, "
            f"on {log.get('date', '').replace('T', ' ')}.<br>"
            f"<b>Held-out balanced accuracy: {accuracy}</b> on {held_out.get('num_units', '?')} units which were not "
            "used for training: the same classifier trained without the new units, then with them."
        )
        if log.get("num_same_as_original", 0) > 0:
            summary += (f"<br>{log['num_same_as_original']} relabelled units were not added: in this model's classes your "
                        "new label was the same as their original label, so they teach it nothing new.")
        if log.get("num_not_in_classes", 0) > 0:
            summary += (f"<br>{log['num_not_in_classes']} relabelled units were not added: their label is not one of "
                        "this model's classes (e.g. noise for a SUA vs MUA model).")
        summary_label = QtWidgets.QLabel(summary)
        summary_label.setWordWrap(True)
        summary_label.setTextFormat(Qt.TextFormat.RichText)
        layout.addWidget(summary_label)

        table = QtWidgets.QTableWidget(len(units), len(UNIT_COLUMNS))
        table.setHorizontalHeaderLabels([UNIT_COLUMN_TITLES[column] for column in UNIT_COLUMNS])
        table.setEditTriggers(QtWidgets.QAbstractItemView.NoEditTriggers)
        table.verticalHeader().setVisible(False)
        table.setSortingEnabled(False)
        for row, (_, unit) in enumerate(units.iterrows()):
            for column, name in enumerate(UNIT_COLUMNS):
                value = short_path(unit[name]) if name == "analyzer" else str(unit[name])
                item = QtWidgets.QTableWidgetItem(value if value != "" else "—")
                if name == "analyzer":
                    item.setToolTip(str(unit[name]))
                # units where you disagreed with the model stand out
                if name == "your_label" and unit["model_label"] not in ("", unit["taught_as"]):
                    item.setForeground(Qt.darkRed)
                table.setItem(row, column, item)
        table.setSortingEnabled(True)
        table.horizontalHeader().setSectionResizeMode(QtWidgets.QHeaderView.ResizeToContents)
        table.horizontalHeader().setStretchLastSection(True)
        layout.addWidget(table, 1)

        layout.addWidget(QtWidgets.QLabel(
            "<span style='color:#667085'><b>Original label</b>: your curated or imported label before relabelling (— if "
            "the unit had none). <b>Model's label</b>: what the model said when the unit was picked. <b>Taught as</b>: "
            "your label in the model's classes, which the model learned. Red: you disagreed with the model.</span>"))

        buttons = QtWidgets.QHBoxLayout()
        buttons.addWidget(QtWidgets.QLabel(f"<span style='color:#667085'>Saved in {Path(folder) / UNITS_FILE}</span>"), 1)
        close_button = theme.make_primary(QtWidgets.QPushButton("Close"))
        close_button.clicked.connect(self.close)
        buttons.addWidget(close_button)
        layout.addLayout(buttons)
        self.resize(900, 600)
