import json
import shutil
from functools import partial
from pathlib import Path

import pandas as pd
import PyQt5.QtWidgets as QtWidgets
from PyQt5.QtCore import pyqtSignal


# UnitRefine models published on the Hugging Face Hub, shown as options when loading from HFH.
# Each group is shown with a gap before it and an optional heading; models are (repo id, note shown after the name).
CASCADE_NOTE = "(can be combined in a cascade for good / MUA / noise labels)"
LIGHTWEIGHT_NOTE = "(based on fast-to-compute metrics, no PCA metrics; can be combined in a cascade for good / MUA / noise labels)"
hfh_model_groups = [
    dict(title=None, models=[
        ("SpikeInterface/UnitRefine_noise_neural_classifier", CASCADE_NOTE),
        ("SpikeInterface/UnitRefine_sua_mua_classifier", CASCADE_NOTE),
    ]),
    dict(title=None, models=[
        ("SpikeInterface/UnitRefine_noise_neural_classifier_lightweight", LIGHTWEIGHT_NOTE),
        ("SpikeInterface/UnitRefine_sua_mua_classifier_lightweight", LIGHTWEIGHT_NOTE),
    ]),
    dict(title="SUA only models", models=[
        ("AnoushkaJain3/UnitRefine-generalized-sua-classifier", ""),
        ("AnoushkaJain3/UnitRefine-mice-sua-classifier", ""),
        ("AnoushkaJain3/UnitRefine-rat-sua-classifier", ""),
        ("AnoushkaJain3/UnitRefine-mole-rat-sua-classifier", ""),
        ("AnoushkaJain3/UnitRefine-monkey-sua-classifier", ""),
        ("AnoushkaJain3/UnitRefine-human-sua-classifier", ""),
    ]),
]


class HFHModelsDialog(QtWidgets.QDialog):
    """Lets the user pick one or more Hugging Face Hub models to download into the project."""

    def __init__(self, existing_repo_ids, parent=None):

        super().__init__(parent)

        self.setWindowTitle("Load models from Hugging Face Hub")
        self.setMinimumWidth(550)

        layout = QtWidgets.QVBoxLayout(self)
        layout.addWidget(QtWidgets.QLabel("Select the UnitRefine models to download:"))

        self.checkboxes = {}
        for group_index, group in enumerate(hfh_model_groups):
            if group_index > 0:
                layout.addSpacing(15)
            if group["title"] is not None:
                group_title = QtWidgets.QLabel(group["title"])
                group_title.setStyleSheet("font-weight: bold;")
                layout.addWidget(group_title)

            for repo_id, note in group["models"]:
                checkbox = QtWidgets.QCheckBox(f"{repo_id}  {note}".strip())
                if repo_id in existing_repo_ids:
                    checkbox.setText(f"{repo_id}  (already in project)")
                    checkbox.setChecked(True)
                    checkbox.setEnabled(False)
                self.checkboxes[repo_id] = checkbox
                layout.addWidget(checkbox)

        other_label = QtWidgets.QLabel("Other Hugging Face repos (one per line, e.g. username/model-name):")
        other_label.setStyleSheet("font-weight: bold;")
        layout.addWidget(other_label)

        self.other_repos_form = QtWidgets.QPlainTextEdit()
        self.other_repos_form.setFixedHeight(70)
        layout.addWidget(self.other_repos_form)

        button_box = QtWidgets.QDialogButtonBox(
            QtWidgets.QDialogButtonBox.StandardButton.Ok | QtWidgets.QDialogButtonBox.StandardButton.Cancel
        )
        button_box.accepted.connect(self.accept)
        button_box.rejected.connect(self.reject)
        layout.addWidget(button_box)

    def get_repo_ids(self):
        """Returns the selected repo ids which are not already in the project, without duplicates."""

        repo_ids = [repo_id for repo_id, checkbox in self.checkboxes.items() if checkbox.isChecked() and checkbox.isEnabled()]
        for line in self.other_repos_form.toPlainText().replace(",", "\n").splitlines():
            repo_id = line.strip()
            if repo_id != "" and repo_id not in repo_ids:
                repo_ids.append(repo_id)
        return repo_ids


def get_model_summary(model_folder):
    """
    Returns the classes a model predicts, the number of training units per class and the
    best balanced accuracy, read from the files saved next to the model.
    """

    model_folder = Path(model_folder)
    summary = {"classes": [], "class_counts": {}, "balanced_accuracy": None}

    model_info_path = model_folder / "model_info.json"
    if model_info_path.is_file():
        with open(model_info_path) as f:
            label_conversion = json.load(f).get("label_conversion", {})
        summary["classes"] = [label_conversion[key] for key in sorted(label_conversion, key=int)]

        labels_path = model_folder / "labels.csv"
        if labels_path.is_file():
            labels = pd.read_csv(labels_path)
            label_column = [column for column in labels.columns if column != "unit_index"][0]
            # labels are stored as class indices; retrained models can store them as floats
            class_indices = pd.to_numeric(labels[label_column], errors="coerce").dropna().astype(int).astype(str)
            counts = class_indices.map(label_conversion).value_counts()
            summary["class_counts"] = {label: int(counts.get(label, 0)) for label in summary["classes"]}

    accuracies_path = model_folder / "model_accuracies.csv"
    if accuracies_path.is_file():
        accuracies = pd.read_csv(accuracies_path)
        # older models saved the balanced accuracy in an `accuracy` column
        accuracy_column = "balanced_accuracy" if "balanced_accuracy" in accuracies.columns else "accuracy"
        if accuracy_column in accuracies.columns and len(accuracies) > 0:
            summary["balanced_accuracy"] = float(accuracies[accuracy_column].max())

    return summary


class ManageModelsWindow(QtWidgets.QDialog):
    """Lists the models in a project with the classes they were trained on, and lets the user delete them."""

    models_changed = pyqtSignal()

    def __init__(self, project, parent=None):

        super().__init__(parent)

        self.project = project

        self.setWindowTitle("UNITREFINE: Manage models")
        self.resize(900, 400)

        layout = QtWidgets.QVBoxLayout(self)
        layout.addWidget(QtWidgets.QLabel("Models in this project. Balanced accuracy is on the held-out test set."))

        self.table = QtWidgets.QTableWidget()
        self.table.setColumnCount(5)
        self.table.setHorizontalHeaderLabels(["Model", "Source", "Classes (training units)", "Balanced accuracy", ""])
        self.table.setEditTriggers(QtWidgets.QAbstractItemView.NoEditTriggers)
        self.table.verticalHeader().setVisible(False)
        layout.addWidget(self.table)

        self.make_table()

    def make_table(self):

        self.table.setRowCount(len(self.project.model_paths))

        for row, (model_folder, hfh_or_local) in enumerate(self.project.model_paths):

            summary = get_model_summary(model_folder)

            if len(summary["class_counts"]) > 0:
                classes_text = ", ".join(f"{label}: {count}" for label, count in summary["class_counts"].items())
            else:
                classes_text = ", ".join(summary["classes"]) if len(summary["classes"]) > 0 else "unknown"

            accuracy = summary["balanced_accuracy"]
            accuracy_text = f"{accuracy:.3f}" if accuracy is not None else "---"

            source_text = "Hugging Face" if hfh_or_local == "hfh" else "local"

            model_item = QtWidgets.QTableWidgetItem(Path(model_folder).name)
            model_item.setToolTip(str(model_folder))
            self.table.setItem(row, 0, model_item)
            self.table.setItem(row, 1, QtWidgets.QTableWidgetItem(source_text))
            self.table.setItem(row, 2, QtWidgets.QTableWidgetItem(classes_text))
            self.table.setItem(row, 3, QtWidgets.QTableWidgetItem(accuracy_text))

            delete_button = QtWidgets.QToolButton()
            delete_button.setIcon(self.style().standardIcon(QtWidgets.QStyle.SP_TrashIcon))
            delete_button.setToolTip(f"Delete {Path(model_folder).name}")
            delete_button.clicked.connect(partial(self.delete_model, model_folder))
            self.table.setCellWidget(row, 4, delete_button)

        self.table.resizeColumnsToContents()

    def model_is_in_project_folder(self, model_folder):
        return Path(model_folder).resolve().parent == (Path(self.project.folder_name) / "models").resolve()

    def delete_model(self, model_folder):

        model_name = Path(model_folder).name
        in_project = self.model_is_in_project_folder(model_folder)

        if in_project:
            message = f"Delete the model '{model_name}'?\n\nThis permanently deletes the folder:\n{model_folder}"
        else:
            message = (
                f"Remove the model '{model_name}' from this project?\n\n"
                f"It was loaded from outside the project, so its folder is not deleted:\n{model_folder}"
            )

        answer = QtWidgets.QMessageBox.question(self, "Delete model", message)
        if answer != QtWidgets.QMessageBox.Yes:
            return

        if in_project:
            shutil.rmtree(model_folder)
            print(f"Deleted model at {model_folder}.")
        else:
            print(f"Removed model at {model_folder} from project.")

        self.project.model_paths = [model for model in self.project.model_paths if Path(model[0]) != Path(model_folder)]

        self.make_table()
        self.models_changed.emit()
