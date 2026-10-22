from pathlib import Path

import PyQt5.QtWidgets as QtWidgets
from spikeinterface_gui.main import check_folder_is_analyzer


def find_analyzers(folder, max_depth=3):
    """Finds SortingAnalyzer folders inside `folder`, without looking inside the analyzers themselves."""

    found = []
    folder = Path(folder)
    if check_folder_is_analyzer(str(folder)):
        return [folder]
    if max_depth == 0:
        return found
    try:
        subfolders = sorted(path for path in folder.iterdir() if path.is_dir() and not path.name.startswith("."))
    except PermissionError:
        return found
    for subfolder in subfolders:
        found += find_analyzers(subfolder, max_depth - 1)
    return found


class ApplyModelsDialog(QtWidgets.QDialog):
    """Pick the analyzers to apply the selected models to, and the folder to save the predictions in."""

    def __init__(self, selection_text, default_output_folder, parent=None):

        super().__init__(parent)

        self.setWindowTitle("UNITREFINE: Apply models to analyzers")
        self.setMinimumWidth(650)

        layout = QtWidgets.QVBoxLayout(self)

        selection_label = QtWidgets.QLabel(f"Models: {selection_text}")
        selection_label.setWordWrap(True)
        layout.addWidget(selection_label)

        analyzers_label = QtWidgets.QLabel("Analyzers the models will be applied to:")
        analyzers_label.setStyleSheet("font-weight: bold;")
        layout.addWidget(analyzers_label)

        self.analyzer_list = QtWidgets.QListWidget()
        self.analyzer_list.setSelectionMode(QtWidgets.QAbstractItemView.ExtendedSelection)
        layout.addWidget(self.analyzer_list)

        buttons_layout = QtWidgets.QHBoxLayout()
        add_button = QtWidgets.QPushButton("+ Add analyzer folder")
        add_button.clicked.connect(self.add_analyzer_folder)
        add_all_button = QtWidgets.QPushButton("+ Add all analyzers inside a folder")
        add_all_button.setToolTip("Searches up to 3 folder levels down for Sorting Analyzer folders")
        add_all_button.clicked.connect(self.add_analyzers_inside_folder)
        remove_button = QtWidgets.QPushButton("Remove selected")
        remove_button.clicked.connect(self.remove_selected)
        buttons_layout.addWidget(add_button)
        buttons_layout.addWidget(add_all_button)
        buttons_layout.addWidget(remove_button)
        layout.addLayout(buttons_layout)

        output_label = QtWidgets.QLabel("Save the predictions (one csv per analyzer) in:")
        output_label.setStyleSheet("font-weight: bold;")
        layout.addWidget(output_label)

        output_layout = QtWidgets.QHBoxLayout()
        self.output_folder_form = QtWidgets.QLineEdit(str(default_output_folder))
        browse_button = QtWidgets.QPushButton("Browse")
        browse_button.clicked.connect(self.choose_output_folder)
        output_layout.addWidget(self.output_folder_form)
        output_layout.addWidget(browse_button)
        layout.addLayout(output_layout)

        button_box = QtWidgets.QDialogButtonBox()
        self.apply_button = button_box.addButton("Apply", QtWidgets.QDialogButtonBox.AcceptRole)
        button_box.addButton(QtWidgets.QDialogButtonBox.StandardButton.Cancel)
        button_box.accepted.connect(self.accept_if_ready)
        button_box.rejected.connect(self.reject)
        layout.addWidget(button_box)

    def add_paths(self, paths):
        existing = set(self.analyzer_paths())
        for path in paths:
            if str(path) not in existing:
                self.analyzer_list.addItem(str(path))
                existing.add(str(path))

    def add_analyzer_folder(self):
        folder = QtWidgets.QFileDialog.getExistingDirectory(self, "Select a Sorting Analyzer folder")
        if folder == "":
            return
        if check_folder_is_analyzer(folder):
            self.add_paths([Path(folder)])
        else:
            print(f"Selected directory {folder} is not a SortingAnalyzer.")

    def add_analyzers_inside_folder(self):
        folder = QtWidgets.QFileDialog.getExistingDirectory(self, "Select a folder containing Sorting Analyzers")
        if folder == "":
            return
        analyzers = find_analyzers(folder)
        print(f"Found {len(analyzers)} Sorting Analyzers in {folder}.")
        self.add_paths(analyzers)

    def remove_selected(self):
        for item in self.analyzer_list.selectedItems():
            self.analyzer_list.takeItem(self.analyzer_list.row(item))

    def choose_output_folder(self):
        folder = QtWidgets.QFileDialog.getExistingDirectory(self, "Select the folder to save the predictions in")
        if folder != "":
            self.output_folder_form.setText(folder)

    def analyzer_paths(self):
        return [self.analyzer_list.item(row).text() for row in range(self.analyzer_list.count())]

    def output_folder(self):
        return self.output_folder_form.text().strip()

    def accept_if_ready(self):
        if len(self.analyzer_paths()) == 0:
            QtWidgets.QMessageBox.warning(self, "No analyzers", "Add at least one analyzer to apply the models to.")
            return
        if self.output_folder() == "":
            QtWidgets.QMessageBox.warning(self, "No output folder", "Choose a folder to save the predictions in.")
            return
        self.accept()
