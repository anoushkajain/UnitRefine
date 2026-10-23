from pathlib import Path

import PyQt5.QtWidgets as QtWidgets
from PyQt5.QtCore import Qt, pyqtSignal

from unitrefine.predictions import get_model_classes, is_layer1_model, is_layer2_model, selection_key


class ModelSelectionWidget(QtWidgets.QWidget):
    """
    Choose the models to inspect, relabel and retrain:
        - Binary: one or more models, each applied on its own
        - Cascade: a noise vs neural model (layer 1), then a SUA vs MUA model (layer 2) on the neural units
    """

    selection_changed = pyqtSignal()

    def __init__(self, parent=None):

        super().__init__(parent)

        self.model_paths = []

        layout = QtWidgets.QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)

        mode_layout = QtWidgets.QHBoxLayout()
        self.binary_radio = QtWidgets.QRadioButton("Each model separately")
        self.cascade_radio = QtWidgets.QRadioButton("As a cascade (noise vs neural, then SUA vs MUA on the neural units)")
        self.binary_radio.setChecked(True)
        self.binary_radio.toggled.connect(self.on_mode_changed)
        mode_layout.addWidget(self.binary_radio)
        mode_layout.addWidget(self.cascade_radio)
        mode_layout.addStretch()
        layout.addLayout(mode_layout)

        self.binary_list = QtWidgets.QListWidget()
        # fixed, so that on short screens the page scrolls instead of the list being squeezed
        self.binary_list.setFixedHeight(140)
        self.binary_list.itemChanged.connect(lambda _: self.selection_changed.emit())
        layout.addWidget(self.binary_list)

        self.cascade_widget = QtWidgets.QWidget()
        cascade_layout = QtWidgets.QFormLayout(self.cascade_widget)
        cascade_layout.setContentsMargins(0, 0, 0, 0)
        self.layer1_combo = QtWidgets.QComboBox()
        self.layer2_combo = QtWidgets.QComboBox()
        self.layer1_combo.currentIndexChanged.connect(lambda _: self.selection_changed.emit())
        self.layer2_combo.currentIndexChanged.connect(lambda _: self.selection_changed.emit())
        cascade_layout.addRow("Layer 1 (noise vs neural):", self.layer1_combo)
        cascade_layout.addRow("Layer 2 (SUA vs MUA):", self.layer2_combo)
        layout.addWidget(self.cascade_widget)
        self.cascade_widget.setVisible(False)

    def set_models(self, model_paths):
        """
        Rebuilds the lists from the project's models. Which binary models are ticked:
            - first time: the last model
            - after models were trained or loaded: the new models
            - otherwise: the same models as before
        New models which fit a cascade layer are also chosen for that layer.
        """

        previously_checked = set(self.checked_binary_names())
        previous_names = [Path(model_folder).name for model_folder, _ in self.model_paths]
        previous_layers = [self.layer1_combo.currentText(), self.layer2_combo.currentText()]

        self.model_paths = list(model_paths)
        names = [Path(model_folder).name for model_folder, _ in self.model_paths]
        new_names = [name for name in names if name not in previous_names]

        if len(previous_names) == 0:
            names_to_check = set(names[-1:])
        elif len(new_names) > 0:
            names_to_check = set(new_names)
        else:
            names_to_check = previously_checked

        for widget in [self, self.binary_list, self.layer1_combo, self.layer2_combo]:
            widget.blockSignals(True)

        self.binary_list.clear()
        self.layer1_combo.clear()
        self.layer2_combo.clear()

        for model_folder, _ in self.model_paths:
            name = Path(model_folder).name
            classes = get_model_classes(model_folder)

            item = QtWidgets.QListWidgetItem(f"{name}   [{', '.join(classes)}]")
            item.setData(Qt.UserRole, name)
            item.setFlags(item.flags() | Qt.ItemIsUserCheckable)
            item.setCheckState(Qt.Checked if name in names_to_check else Qt.Unchecked)
            self.binary_list.addItem(item)

            if is_layer1_model(classes):
                self.layer1_combo.addItem(name)
            if is_layer2_model(classes):
                self.layer2_combo.addItem(name)

        for combo, previous in zip([self.layer1_combo, self.layer2_combo], previous_layers):
            combo_names = [combo.itemText(index) for index in range(combo.count())]
            new_for_layer = [name for name in new_names if name in combo_names]
            choice = new_for_layer[-1] if len(new_for_layer) > 0 else previous
            if choice in combo_names:
                combo.setCurrentIndex(combo_names.index(choice))

        for widget in [self, self.binary_list, self.layer1_combo, self.layer2_combo]:
            widget.blockSignals(False)

        self.selection_changed.emit()

    def on_mode_changed(self):
        is_binary = self.binary_radio.isChecked()
        self.binary_list.setVisible(is_binary)
        self.cascade_widget.setVisible(not is_binary)
        self.selection_changed.emit()

    def mode(self):
        return "binary" if self.binary_radio.isChecked() else "cascade"

    def checked_binary_names(self):
        return [
            self.binary_list.item(row).data(Qt.UserRole)
            for row in range(self.binary_list.count())
            if self.binary_list.item(row).checkState() == Qt.Checked
        ]

    def selected_model_names(self):
        if self.mode() == "binary":
            return self.checked_binary_names()
        layer1, layer2 = self.layer1_combo.currentText(), self.layer2_combo.currentText()
        if layer1 == "" or layer2 == "":
            return []
        return [layer1, layer2]

    def selected_models(self):
        """Returns [(model_folder, hfh_or_local), ...]; for a cascade, [layer 1, layer 2]."""
        models_by_name = {Path(model_folder).name: (Path(model_folder), hfh_or_local) for model_folder, hfh_or_local in self.model_paths}
        return [models_by_name[name] for name in self.selected_model_names()]

    def selection_key(self):
        names = self.selected_model_names()
        if len(names) == 0:
            return None
        return selection_key(self.mode(), names)

    def describe_problem(self):
        """Returns why the selection can't be used, or None if it can."""
        if len(self.model_paths) == 0:
            return "No models trained or loaded into project! Please train or load one first."
        if self.mode() == "binary" and len(self.checked_binary_names()) == 0:
            return "No model selected. Tick at least one model."
        if self.mode() == "cascade":
            if self.layer1_combo.count() == 0:
                return "No noise vs neural model in the project for cascade layer 1."
            if self.layer2_combo.count() == 0:
                return "No SUA vs MUA model in the project for cascade layer 2."
        return None
