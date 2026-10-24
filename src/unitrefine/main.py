import json
import subprocess
import sys
import shutil
import os

from pathlib import Path
from argparse import ArgumentParser
from functools import partial

import numpy as np
import pandas as pd

from huggingface_hub import repo_exists
from sklearn.base import clone
from sklearn.metrics import balanced_accuracy_score
from spikeinterface.curation.model_based_curation import load_model
from skops.io import dump

import PyQt5.QtWidgets as QtWidgets
from PyQt5.QtCore import Qt, QSize, QProcess, QProcessEnvironment, QTimer, QCoreApplication, QEvent, QUrl
from PyQt5.QtGui import QDesktopServices, QFont, QFontDatabase, QIcon
from PyQt5.QtWidgets import QStyleFactory

from spikeinterface_gui.main import check_folder_is_analyzer
from spikeinterface.core.core_tools import is_path_remote
from spikeinterface.core import load_sorting_analyzer

from unitrefine import LABELS_BACKUP_MARKER, SIGUI_SHOWN_MARKER, SIGUI_STEP_MARKER, unwritable_file_message
from unitrefine.sigui_progress import OpeningDialog, CURATE_STEPS, INSPECT_STEPS
from unitrefine.train import TRAINING_UNITS_FILE, TrainWindow, read_training_units
from unitrefine.import_labels import import_labels, compute_all_metrics
from unitrefine.manage_models import ManageModelsWindow, HFHModelsDialog
from unitrefine.model_selection import ModelSelectionWidget
from unitrefine.predictions import (
    QUALITY_ALIASES,
    analyzer_output_names,
    label_units_function,
    apply_to_analyzers,
    find_least_confident_units,
    map_quality_to_model_class,
    metrics_for_model,
    as_model_input,
    predictions_file_name,
    relabelled_units_path,
    units_to_relabel_path,
)
from unitrefine.apply_models import find_analyzers
from unitrefine.evaluation import read_trained_on, write_trained_on, normalise_path
from unitrefine import figures, theme
from unitrefine.launcher import enable_high_dpi_scaling, set_app_font
from unitrefine.phy_import import (
    EXPORTED_STEPS, GAIN_HINT, KILOSORT_STEPS, PHY_RESULT_MARKER, PHY_STEP_MARKER, default_output_folder, made_from_phy,
    matches_phy_folder, phy_folder_kind, recording_file, recording_needs_gain,
)
from unitrefine.retraining_history import RetrainingHistoryTable, write_log as write_retraining_log
from unitrefine.widgets import (
    BusyNotice,
    CollapsibleSection,
    ElidedLabel,
    FigureGallery,
    FigureWorker,
    PathRow,
    chip,
    clear_layout,
    group_label,
    list_box,
    list_message,
    sigui_help_box,
    muted_label,
    open_folder,
    subheading,
)

import re
_RETRAINED_RE = re.compile(r"^(?P<base>.*?)(?:_retrained(?:_(?P<idx>\d+))?)?$")

import warnings
warnings.filterwarnings("ignore")

UPLOAD_TUTORIAL_URL = "https://spikeinterface.readthedocs.io/en/stable/tutorials/curation/plot_3_upload_a_model.html"

APP_DESCRIPTION = "a machine-learning toolbox that simplifies spike sorting curation by reducing manual curation"

# width of the column holding the sections, in device independent pixels
CONTENT_MIN_WIDTH = 720
CONTENT_MAX_WIDTH = 1100


class UrlInputDialog(QtWidgets.QDialog):
    def __init__(self, parent=None, default=None, background_color=None):
        super().__init__(parent)

        if default is None:
            default = "https://www.example.com"
        
        self.setWindowTitle("Open URL")
        self.setMinimumWidth(400)

        layout = QtWidgets.QVBoxLayout()

        self.label = QtWidgets.QLabel("Please enter the s3 path to an analyzer:")
        layout.addWidget(self.label)

        self.url_input = QtWidgets.QLineEdit()
        self.url_input.setPlaceholderText(default)
        self.url_input.setClearButtonEnabled(True)
        layout.addWidget(self.url_input)

        self.button_box = QtWidgets.QDialogButtonBox(
            QtWidgets.QDialogButtonBox.StandardButton.Ok | QtWidgets.QDialogButtonBox.StandardButton.Cancel
        )
        layout.addWidget(self.button_box)

        self.setLayout(layout)

        self.button_box.accepted.connect(self.accept)
        self.button_box.rejected.connect(self.reject)

    def get_url(self):
        """Returns the text entered in the line edit."""
        return self.url_input.text().strip()

class UnitRefineProject:

    def __init__(self, folder_name):

        self.folder_name = folder_name
        self.analyzers = {}
        self.model_paths = []
        self.config = {}
        self.selected_model = None

    def save(self, folder_name=None):

        if folder_name is None:
            folder_name = self.folder_name

        Path(folder_name).mkdir(exist_ok=True)

        Path(folder_name / "analyzers").mkdir(exist_ok=True)
        Path(folder_name / "models").mkdir(exist_ok=True)

        with open(folder_name / "config.json", 'w') as f:
            json.dump(self.config, f, indent=4)

        for analyzer_name, analyzer_dict in self.analyzers.items():

            path = analyzer_dict.get('path')

            save_path = Path(folder_name / "analyzers" / f"analyzer_{analyzer_name}")
            save_path.mkdir(exist_ok=True)

            if path is not None:
                with open(save_path / 'path.txt', 'w') as output:
                    output.write(str(path))

            # labels.csv is not written here: Curate and Import labels write it themselves. Writing the copy read
            # when the project was opened would replace labels set since then


    def save_config(self):
        """Saves only config.json, leaving the analyzers' label files untouched."""
        config_path = Path(self.folder_name) / "config.json"
        try:
            with open(config_path, 'w') as f:
                json.dump(self.config, f, indent=4)
        except OSError as e:
            # e.g. a read-only file: the settings are not saved, but nothing else is lost
            report_unwritable_file(config_path, e)

    # called with (path, message) when a file can't be written; the main window shows the message
    on_unwritable_file = None

    def add_analyzer(self, directory):

        # a number no analyzer has used, also not one whose folder is still on disk (e.g. left from a removed
        # analyzer), so a new analyzer never picks up another analyzer's files
        used_keys = set(self.analyzers.keys())
        for folder in (Path(self.folder_name) / "analyzers").glob("analyzer_*"):
            if folder.name.split("_")[-1].isdigit():
                used_keys.add(int(folder.name.split("_")[-1]))
        new_key = max(used_keys) + 1 if len(used_keys) > 0 else 0

        analyer_in_project = Path(f'analyzers/analyzer_{new_key}')
        self.analyzers[new_key] = {'path': directory, 'analyzer_in_project': analyer_in_project}


def report_unwritable_file(path, error):
    message = unwritable_file_message(path, error)
    print(message)
    if UnitRefineProject.on_unwritable_file is not None:
        UnitRefineProject.on_unwritable_file(path, message)


def load_project(folder_name):

    folder_name = Path(folder_name)

    project = UnitRefineProject(folder_name)

    config_path = folder_name / "config.json"
    if config_path.is_file():
        with open(config_path) as f:
            project.config = json.load(f)

    analyzers_folder = folder_name / "analyzers"
    # in the order they were added (by number: analyzer_10 after analyzer_9). The folders are listed in no
    # particular order on Linux, and alphabetically on Windows
    analyzer_folders = [folder for folder in analyzers_folder.glob('*/') if re.fullmatch(r"analyzer_\d+", folder.name)]
    analyzer_folders.sort(key=lambda folder: int(folder.name.split('_')[1]))

    for analyzer_folder in analyzer_folders:

        analyzer_dict = {}

        # without path.txt the folder doesn't belong to an analyzer in the project (e.g. it was removed)
        if not (analyzer_folder / 'path.txt').is_file():
            print(f"Skipping {analyzer_folder}: it has no path.txt, so it is not an analyzer in this project.")
            continue

        with open(analyzer_folder / 'path.txt') as f:
            analyzer_path = f.read()
            analyzer_dict['path'] = analyzer_path

        metrics_path = analyzer_folder / 'all_metrics.csv'
        if metrics_path.is_file():
            all_metrics = pd.read_csv(metrics_path)
            analyzer_dict['all_metrics'] = all_metrics

        labelled_metrics_path = analyzer_folder / 'labelled_metrics.csv'
        if labelled_metrics_path.is_file():
            labelled_metrics = pd.read_csv(labelled_metrics_path)
            analyzer_dict['labelled_metrics'] = labelled_metrics

        labels_path = analyzer_folder / 'labels.csv'
        if labels_path.is_file():
            curation = pd.read_csv(labels_path)
            analyzer_dict['labels'] = curation

        analyzer_dict['analyzer_in_project'] = f"analyzers/{Path(analyzer_folder).name}"

        project.analyzers[int(str(analyzer_folder.name).split('_')[1])] = analyzer_dict
            
    models_folder = folder_name / "models"
    model_folders = sorted(folder for folder in models_folder.glob('*/') if '.DS' not in str(folder))

    for model_folder in model_folders:
        if (model_folder / "hfh_path.txt").is_file():
            project.model_paths.append((model_folder, "hfh"))
        else:
            project.model_paths.append((model_folder, "local"))

    return project

class MainWindow(QtWidgets.QWidget):

    # added to the phy_import command (e.g. --n_jobs 1 in the tests)
    phy_import_options = []

    def __init__(self, project):

        super().__init__()

        self.setObjectName("page")

        self.w = None
        self.sorting_analyzer_paths = []

        self.output_folder = project.folder_name

        self.project = project
        self.num_units_to_relabel = 50

        # figures are made one at a time in a background thread; changes are collected for a moment first,
        # so several changes at once (e.g. a new model selected) only update the figures once
        # files which can't be written (e.g. read-only) are reported once each, in a message box
        self.reported_unwritable_files = set()
        UnitRefineProject.on_unwritable_file = self.show_unwritable_file

        self.figure_worker = FigureWorker(self)
        self.figure_refresh_timer = QTimer(self)
        self.figure_refresh_timer.setSingleShot(True)
        self.figure_refresh_timer.setInterval(150)
        self.figure_refresh_timer.timeout.connect(self.refresh_figures)

        # The sections sit in a column centred in a scroll area. The column follows the window width between
        # a minimum (smaller windows scroll sideways) and a maximum (larger windows keep it centred instead of
        # stretching it). Sizes are in device independent pixels, so they follow the screen's display scaling.
        content_widget = QtWidgets.QWidget()
        content_widget.setObjectName("page")
        content_widget.setMinimumWidth(CONTENT_MIN_WIDTH)
        content_widget.setMaximumWidth(CONTENT_MAX_WIDTH)
        self.main_layout = QtWidgets.QVBoxLayout(content_widget)
        self.main_layout.setContentsMargins(4, 20, 4, 28)
        self.main_layout.setSpacing(16)

        centring_widget = QtWidgets.QWidget()
        centring_widget.setObjectName("page")
        centring_layout = QtWidgets.QHBoxLayout(centring_widget)
        centring_layout.setContentsMargins(16, 0, 16, 0)
        # the column takes the width first; the stretches on either side only get what is left once
        # the column reaches its maximum width, which keeps it centred
        centring_layout.addStretch(0)
        centring_layout.addWidget(content_widget, 1, alignment=Qt.AlignmentFlag.AlignTop)
        centring_layout.addStretch(0)

        scroll_area = QtWidgets.QScrollArea()
        scroll_area.setWidgetResizable(True)
        scroll_area.setFrameShape(QtWidgets.QFrame.NoFrame)
        scroll_area.setHorizontalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAsNeeded)
        scroll_area.setVerticalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAsNeeded)
        scroll_area.setWidget(centring_widget)

        window_layout = QtWidgets.QVBoxLayout(self)
        window_layout.setContentsMargins(0, 0, 0, 0)
        window_layout.setSpacing(0)
        # banner shown while the SpikeInterface-GUI is open; outside the scroll area so it's always visible
        self.sigui_banner = QtWidgets.QFrame()
        self.sigui_banner.setStyleSheet(f"QFrame {{ background-color: {theme.WARNING_SOFT}; border-bottom: 1px solid #F5D78E; }}")
        banner_layout = QtWidgets.QHBoxLayout(self.sigui_banner)
        banner_layout.setContentsMargins(20, 10, 20, 10)
        self.sigui_banner_text = QtWidgets.QLabel("")
        self.sigui_banner_text.setWordWrap(True)
        self.sigui_banner_text.setStyleSheet("border: none;")
        banner_layout.addWidget(self.sigui_banner_text, 1)
        self.sigui_banner.setVisible(False)
        window_layout.addWidget(self.sigui_banner)

        window_layout.addWidget(scroll_area)
        self.content_widget = content_widget
        self.sigui_process = None
        # adding a Phy folder (+ Phy output)
        self.phy_process = None
        self.phy_reuse = False
        self.retrainedModelNameForm = None

        header_layout = QtWidgets.QHBoxLayout()
        header_layout.setContentsMargins(4, 0, 4, 4)
        app_title = QtWidgets.QLabel("UnitRefine")
        app_title.setStyleSheet("font-weight: bold; font-size: 20pt;")
        header_layout.addWidget(app_title, 0, Qt.AlignmentFlag.AlignTop)
        app_subtitle = muted_label(APP_DESCRIPTION[0].upper() + APP_DESCRIPTION[1:])
        app_subtitle.setStyleSheet("padding-top: 9px; padding-left: 4px;")
        header_layout.addWidget(app_subtitle, 1)
        self.main_layout.addLayout(header_layout)

        ###############
        # 1. PROJECT
        ###############

        self.project_section = self.make_section(1, "Project")
        self.project_section.set_summary(Path(self.output_folder).name)

        project_form = QtWidgets.QGridLayout()
        project_form.setHorizontalSpacing(12)
        project_form.setVerticalSpacing(6)
        project_form.setColumnStretch(1, 1)
        folders = [
            ("Project folder", Path(self.output_folder)),
            ("Models folder", Path(self.output_folder) / "models"),
            ("Figures folder", Path(self.output_folder) / figures.FIGURES_FOLDER),
        ]
        for row, (name, folder) in enumerate(folders):
            project_form.addWidget(muted_label(name, word_wrap=False), row, 0)
            project_form.addWidget(ElidedLabel(str(folder)), row, 1)
            open_button = theme.make_flat(QtWidgets.QPushButton("Open"))
            open_button.clicked.connect(lambda _checked=False, folder=folder: open_folder(folder))
            project_form.addWidget(open_button, row, 2)
        project_form.addWidget(muted_label("Labels", word_wrap=False), len(folders), 0)
        project_form.addWidget(QtWidgets.QLabel("good, MUA, noise"), len(folders), 1)
        project_form.addWidget(muted_label("Contents", word_wrap=False), len(folders) + 1, 0)
        self.project_counts_label = QtWidgets.QLabel("")
        project_form.addWidget(self.project_counts_label, len(folders) + 1, 1, 1, 2)
        self.project_section.body_layout.addLayout(project_form)

        ###############
        # 2. LABEL AND TRAIN
        ###############

        self.curation_section = self.make_section(2, "Label your units and train a model")
        self.curation_section.body_layout.addWidget(muted_label(
            "Label the units of your recordings yourself with <b>Curate</b> (opens the SpikeInterface-GUI), or "
            "<b>Import labels</b> you already have. Then <b>train a model</b> on them.<br>"
            "<b>Import labels</b> reads Phy (cluster_group.tsv, cluster_info.tsv), SortingView (.json), SpikeInterface "
            "curation (.json), or a .csv/.tsv with <i>unit_id</i> and <i>quality</i> columns. Labels good, MUA and "
            "noise are used; others (e.g. unsorted) are skipped.<br>"
            "<b>+ Phy output</b> adds a Kilosort output folder curated in Phy: an analyzer is made from it and its "
            "recording (a few minutes), saved next to the Phy folder, and the labels you set in Phy are imported. A Phy "
            "folder exported from an analyzer works too: Phy's merges and splits are applied to that analyzer."
        ))

        add_buttons_layout = QtWidgets.QHBoxLayout()
        add_buttons_layout.addWidget(QtWidgets.QLabel("Add your spike sorted data:"))
        self.add_sa_button = QtWidgets.QPushButton("+ Sorting Analyzer folder")
        self.add_sa_button.clicked.connect(self.selectDirectoryDialog)
        add_buttons_layout.addWidget(self.add_sa_button)
        self.add_s3_button = QtWidgets.QPushButton("+ Analyzer from s3")
        self.add_s3_button.clicked.connect(self.add_from_s3)
        add_buttons_layout.addWidget(self.add_s3_button)
        add_phy_button = QtWidgets.QPushButton("+ Phy output")
        add_phy_button.setToolTip("A Kilosort output folder curated in Phy, or a Phy folder exported from an analyzer. "
                                  "An analyzer is made from it, and Phy's labels are imported.")
        add_phy_button.clicked.connect(lambda: self.add_phy_output(for_training=True))
        add_buttons_layout.addWidget(add_phy_button)
        add_buttons_layout.addStretch()
        self.curation_section.body_layout.addLayout(add_buttons_layout)

        self.analyzer_rows_layout = self.add_rows_container(self.curation_section)
        self.curation_section.body_layout.addWidget(sigui_help_box())

        train_layout = QtWidgets.QHBoxLayout()
        train_button = theme.make_primary(QtWidgets.QPushButton("Train a model"))
        train_button.clicked.connect(self.show_train_window)
        train_layout.addWidget(train_button)
        train_layout.addStretch()
        self.curation_section.body_layout.addLayout(train_layout)

        self.training_gallery = FigureGallery(self.figure_worker, self.figures_folder("2_labels_and_training"))
        self.curation_section.body_layout.addWidget(self.training_gallery)

        ###############
        # 3. APPLY MODELS
        ###############

        self.apply_section = self.make_section(3, "Apply models")
        self.apply_section.body_layout.addWidget(muted_label(
            "Get labels from a model you trained or downloaded. Use any model on its own, or combine a noise vs "
            "neural model and a SUA vs MUA model into a <b>cascade</b>. The figures show what the models predict, "
            "and how they compare with your labels."
        ))

        self.apply_section.body_layout.addWidget(subheading("3.1 Load models"))
        load_layout = QtWidgets.QHBoxLayout()
        load_model_button = QtWidgets.QPushButton("+ Load")
        load_model_button.setToolTip("Add a model folder from your computer")
        load_model_button.clicked.connect(self.selectModelDialog)
        load_layout.addWidget(load_model_button)
        load_model_hf = QtWidgets.QPushButton("+ Load from HFH")
        load_model_hf.setToolTip("Download UnitRefine models from the Hugging Face Hub")
        load_model_hf.clicked.connect(self.add_from_hfh)
        load_layout.addWidget(load_model_hf)
        manage_models_button = QtWidgets.QPushButton("Manage models")
        manage_models_button.setToolTip("See the classes each model was trained on, and delete models")
        manage_models_button.clicked.connect(self.show_manage_models_window)
        load_layout.addWidget(manage_models_button)
        load_layout.addWidget(muted_label("Models trained in section 2 are added here.", word_wrap=False), 1)
        self.apply_section.body_layout.addLayout(load_layout)

        self.apply_section.body_layout.addWidget(subheading("3.2 Choose your models"))
        self.apply_section.body_layout.addWidget(muted_label(
            "Tick several models to compare them on the same recordings: their labels and confidence appear side by "
            "side in <b>Inspect</b> and in the figures below."))
        self.model_selector = ModelSelectionWidget(self)
        self.apply_section.body_layout.addWidget(self.model_selector)

        self.apply_section.body_layout.addWidget(subheading("3.3 Apply models to get predictions and inspect"))
        new_data_layout = QtWidgets.QHBoxLayout()
        new_data_layout.addWidget(QtWidgets.QLabel("Add spike sorted data to get predictions on:"))
        add_new_folder_button = QtWidgets.QPushButton("+ Sorting Analyzer folder")
        add_new_folder_button.setToolTip("One analyzer, or a folder with several (searched 3 levels down)")
        add_new_folder_button.clicked.connect(self.add_new_analyzers_from_folder)
        new_data_layout.addWidget(add_new_folder_button)
        add_new_s3_button = QtWidgets.QPushButton("+ Analyzer from s3")
        add_new_s3_button.clicked.connect(self.add_new_analyzer_from_s3)
        new_data_layout.addWidget(add_new_s3_button)
        add_new_phy_button = QtWidgets.QPushButton("+ Phy output")
        add_new_phy_button.setToolTip("A Kilosort output folder curated in Phy, or a Phy folder exported from an analyzer. "
                                      "An analyzer is made from it, and Phy's labels are imported to compare the models with.")
        add_new_phy_button.clicked.connect(lambda: self.add_phy_output(for_training=False))
        new_data_layout.addWidget(add_new_phy_button)
        new_data_layout.addStretch()
        self.apply_section.body_layout.addLayout(new_data_layout)

        # the new analyzers, and (when shown) the training analyzers; rebuilt when they change
        validate_container = QtWidgets.QWidget()
        self.validate_rows_layout = QtWidgets.QVBoxLayout(validate_container)
        self.validate_rows_layout.setContentsMargins(0, 0, 0, 0)
        self.validate_rows_layout.setSpacing(8)
        self.apply_section.body_layout.addWidget(validate_container)
        self.show_training_analyzers = False

        apply_code_button = theme.make_flat(QtWidgets.QPushButton("Generate code to apply the model(s) in a script"))
        apply_code_button.setToolTip("Prints Python code to the terminal which applies the chosen models to an analyzer")
        apply_code_button.clicked.connect(self.make_apply_code)
        code_layout = QtWidgets.QHBoxLayout()
        code_layout.addWidget(apply_code_button)
        code_layout.addStretch()
        self.apply_section.body_layout.addLayout(code_layout)

        self.apply_gallery = FigureGallery(self.figure_worker, self.figures_folder("3_predictions"))
        self.apply_section.body_layout.addWidget(self.apply_gallery)

        #################
        # 4. ACTIVE LEARNING
        ##################

        self.relabel_section = self.make_section(4, "Improve the model: relabel and retrain")
        self.relabel_section.body_layout.addWidget(muted_label(
            "<b>Find</b> the units the model is least sure about across all your analyzers, <b>relabel</b> them in "
            "the SpikeInterface-GUI, and <b>retrain</b>. Units you have already labelled are never picked again, so "
            "each round asks about new units.<br>"
            "<i>Suggestion:</i> repeat for a few rounds, and stop when the <i>Accuracy over rounds</i> figure levels off."
        ))

        self.relabel_section.body_layout.addWidget(subheading("4.1 Find the least confident units"))
        find_layout = QtWidgets.QHBoxLayout()
        self.unitLabel = QtWidgets.QLabel("Number of units to relabel, across all analyzers:")
        find_layout.addWidget(self.unitLabel)
        self.numUnitsForm = QtWidgets.QLineEdit(f"{self.num_units_to_relabel}")
        self.numUnitsForm.setFixedWidth(60)
        self.numUnitsForm.textChanged.connect(self.update_num_units_to_relabel)
        find_layout.addWidget(self.numUnitsForm)
        find_units_button = QtWidgets.QPushButton("Find least confident units")
        find_units_button.setToolTip("Ranks the units of all analyzers by the confidence of the selected model(s), "
                                     "and picks the least confident ones you have not labelled yet")
        find_units_button.clicked.connect(self.pick_units_to_relabel)
        find_layout.addWidget(find_units_button)
        find_layout.addStretch()
        self.relabel_section.body_layout.addLayout(find_layout)

        self.relabel_section.body_layout.addWidget(subheading("4.2 Relabel them in each analyzer"))
        self.relabel_rows_layout = self.add_rows_container(self.relabel_section)

        self.relabel_section.body_layout.addWidget(subheading("4.3 Retrain"))
        retrain_choice_layout = QtWidgets.QHBoxLayout()
        retrain_choice_layout.addWidget(QtWidgets.QLabel("Model to retrain:"))
        self.retrain_choice = QtWidgets.QComboBox()
        self.retrain_choice.setToolTip("With several models chosen in 3.2, retrain one of them, or all")
        self.retrain_choice.currentIndexChanged.connect(self.on_retrain_choice_changed)
        retrain_choice_layout.addWidget(self.retrain_choice, 1)
        self.relabel_section.body_layout.addLayout(retrain_choice_layout)
        self.relabel_section.body_layout.addWidget(muted_label(
            "Choose a model from the models you selected in <b>3.2</b>."))
        retrain_layout = QtWidgets.QHBoxLayout()
        retrain_layout.addWidget(QtWidgets.QLabel("Retrained model name:"))
        self.retrainedModelNameForm = QtWidgets.QLineEdit("")
        retrain_layout.addWidget(self.retrainedModelNameForm, 1)
        retrain_button = theme.make_primary(QtWidgets.QPushButton('Retrain'))
        retrain_button.clicked.connect(partial(self.retrain_model))
        retrain_layout.addWidget(retrain_button)
        self.relabel_section.body_layout.addLayout(retrain_layout)

        self.relabel_section.body_layout.addWidget(subheading("4.4 Retraining history"))
        self.history_message = muted_label("")
        self.relabel_section.body_layout.addWidget(self.history_message)
        self.history_table = RetrainingHistoryTable()
        self.relabel_section.body_layout.addWidget(self.history_table)

        self.active_learning_gallery = FigureGallery(self.figure_worker, self.figures_folder("4_active_learning"))
        self.relabel_section.body_layout.addWidget(self.active_learning_gallery)

        ###############
        # 5. SHARE
        ###############

        self.share_section = self.make_section(5, "Share your model")
        self.share_section.body_layout.addWidget(muted_label(
            "You can upload your models to the Hugging Face Hub, so others can download and use them, like the "
            "UnitRefine models in <b>+ Load from HFH</b>."))
        tutorial_layout = QtWidgets.QHBoxLayout()
        tutorial_button = theme.make_flat(QtWidgets.QPushButton("Optional: open the tutorial on uploading a model"))
        tutorial_button.setToolTip(UPLOAD_TUTORIAL_URL)
        tutorial_button.clicked.connect(lambda: QDesktopServices.openUrl(QUrl(UPLOAD_TUTORIAL_URL)))
        tutorial_layout.addWidget(tutorial_button)
        tutorial_layout.addStretch()
        self.share_section.body_layout.addLayout(tutorial_layout)

        self.main_layout.addStretch(1)

        self.make_curation_button_list()
        self.make_relabel_button_list()

        # fill the model selection once all the sections it updates exist
        self.model_selector.selection_changed.connect(self.on_model_selection_changed)
        self.make_model_list()

    def finish_loading(self):
        """
        Fills in everything before the window is first shown, so it doesn't change in front of the user: the
        figures (saved ones are shown at once, the others are made in the background), and the widgets which
        were replaced while the lists were built.
        """
        self.figure_refresh_timer.stop()
        self.refresh_figures()
        QCoreApplication.sendPostedEvents(None, QEvent.DeferredDelete)
        self.ensurePolished()
        self.layout().activate()

    def make_section(self, number, title):
        """A collapsible section card; whether it is folded is remembered in the project's config."""
        collapsed = self.project.config.get("collapsed_sections")
        if collapsed is None:
            # a new project opens with section 3 only: applying a (downloaded) model is where most people start
            expanded = number == 3
        else:
            # the project details are folded unless opened: they are not needed all the time
            expanded = not collapsed.get(str(number), number == 1)
        section = CollapsibleSection(number, title, expanded=expanded)
        section.toggled.connect(lambda expanded, number=number: self.remember_section_state(number, expanded))
        self.main_layout.addWidget(section)
        return section

    def remember_section_state(self, number, expanded):
        self.project.config.setdefault("collapsed_sections", {})[str(number)] = not expanded
        self.project.save_config()

    def add_rows_container(self, section):
        """A list of rows (e.g. one per analyzer) which is rebuilt when the project changes."""
        container, rows_layout = list_box()
        section.body_layout.addWidget(container)
        return rows_layout

    def figures_folder(self, section_folder):
        return Path(self.output_folder) / figures.FIGURES_FOLDER / section_folder

    def schedule_figure_refresh(self):
        self.figure_refresh_timer.start()

    def refresh_figures(self):
        """
        Updates the figures of sections 2, 3 and 4. Figures whose files did not change are shown at once
        from the saved png; the others are made again in the background.
        """
        targets = self.relabel_targets()

        # until some units are labelled, there is nothing to plot: say so instead of showing empty figures
        has_labels = any(
            (Path(target['save_folder']) / "labels.csv").is_file() for target in targets if target['analyzer_index'] is not None
        )
        if has_labels:
            self.training_gallery.set_specs(figures.training_specs(self.output_folder, targets, self.trained_model_folders()))
        elif len(self.project.analyzers) == 0:
            self.training_gallery.set_specs([], empty_message="Figures appear here once you add your spike sorted data and label its units.")
        else:
            self.training_gallery.set_specs([], empty_message="Figures appear here once you curate or import labels.")

        problem = self.model_selector.describe_problem()
        if problem is not None:
            message = f"Figures appear here once models are chosen. {problem}"
            self.apply_gallery.set_specs([], empty_message=message)
            self.active_learning_gallery.set_specs([], empty_message=message)
            return

        mode = self.model_selector.mode()
        model_folders = [model_folder for model_folder, _ in self.model_selector.selected_models()]
        key = self.model_selector.selection_key()
        apply_specs = figures.apply_specs(self.output_folder, targets, mode, model_folders, key)
        self.apply_gallery.set_specs(apply_specs, folder=apply_specs[0].path.parent)
        lineages = dict(self.model_lineage(model_folder.name) for model_folder in model_folders)
        active_learning_specs = figures.active_learning_specs(self.output_folder, targets, mode, model_folders, key, lineages)
        self.active_learning_gallery.set_specs(active_learning_specs, folder=active_learning_specs[0].path.parent)

    def trained_model_folders(self, max_models=6):
        """The models trained in this project (not retrained versions), newest first, for the figures of section 2."""
        trained = []
        for model_folder, hfh_or_local in self.project.model_paths:
            model_folder = Path(model_folder)
            base_name, _ = _base_and_idx(model_folder.name)
            if hfh_or_local != "local" or base_name != model_folder.name:
                continue
            if not ((model_folder / figures.TEST_PREDICTIONS_FILE).is_file() or (model_folder / "trained_on.json").is_file()):
                continue
            model_file = model_folder / "best_model.skops"
            if model_file.is_file():
                trained.append((model_file.stat().st_mtime, model_folder))
        return [model_folder for _, model_folder in sorted(trained, key=lambda item: item[0], reverse=True)][:max_models]

    def update_summaries(self):
        """The short summaries in the section headers, and the counts in the project details."""
        num_labelled = 0
        for analyzer in self.project.analyzers.values():
            labels_path = Path(self.project.folder_name) / analyzer['analyzer_in_project'] / "labels.csv"
            if labels_path.is_file():
                num_labelled += len(pd.read_csv(labels_path))
        num_analyzers = len(self.project.analyzers)
        num_applied = len(self.project.config.get("applied_analyzers", []))
        self.project_counts_label.setText(
            f"{num_analyzers} analyzer{'s' if num_analyzers != 1 else ''}, {num_labelled} labelled units, "
            f"{len(self.project.model_paths)} models, {num_applied} new analyzers with predictions"
        )
        self.curation_section.set_summary(
            f"{num_analyzers} analyzer{'s' if num_analyzers != 1 else ''} · {num_labelled} labelled units"
        )
        if hasattr(self, "model_selector"):
            names = self.model_selector.selected_model_names()
            self.apply_section.set_summary(self.selection_description() if len(names) > 0 else "no model chosen")

    def add_from_s3(self):
        
        dialog = UrlInputDialog()

        if dialog.exec():
            url = dialog.get_url()
            if is_path_remote(url):
                self.project.add_analyzer(url)
                self.project.save()
                self.make_curation_button_list()
                self.make_validate_button_list()
                self.make_relabel_button_list()
                print(f"Successfully added analyzer at {url} to project.")
            else:
                print(f"url {url} is not a valid analyzer path.")

    def update_num_units_to_relabel(self):

        try:
            num_units = int(self.numUnitsForm.text())
        except ValueError:
            print("Cannot parse the number of units to relabel. Must be a whole number. Please change the value.")
            return
        if num_units < 1:
            print("Number of units to relabel should be at least 1. Please change the value.")
            return
        self.num_units_to_relabel = num_units

    def relabel_targets(self):
        """
        The analyzers whose units can be relabelled: the analyzers in the project (section 2) and the
        new analyzers the models were applied to (section 3.3). Each is a dict with the analyzer `path`,
        the `save_folder` and `file_prefix` of its files, and its `analyzer_index` in the project (None for new analyzers).
        """
        targets = []
        for analyzer_index, analyzer in enumerate(self.project.analyzers.values()):
            targets.append(dict(
                path=str(analyzer['path']),
                save_folder=Path(self.project.folder_name) / analyzer['analyzer_in_project'],
                file_prefix="",
                analyzer_index=analyzer_index,
                analyzer=analyzer,
            ))
        for applied in self.project.config.get("applied_analyzers", []):
            targets.append(dict(
                path=applied['path'],
                save_folder=Path(applied['output_folder']),
                file_prefix=f"{applied['output_name']}__",
                analyzer_index=None,
                analyzer={'path': applied['path']},
            ))
        return targets

    def pick_units_to_relabel(self):

        problem = self.model_selector.describe_problem()
        if problem is not None:
            print(problem)
            return

        targets = self.relabel_targets()
        if len(targets) == 0:
            print("No analyzers in the project. Add analyzers in section 2, or apply the models to new analyzers in section 3.3.")
            return

        mode = self.model_selector.mode()
        model_folders = [model_folder for model_folder, _ in self.model_selector.selected_models()]
        selection_key = self.model_selector.selection_key()

        print(f"\nFinding the {self.num_units_to_relabel} least confident units of {self.selection_description()}, across {len(targets)} analyzers...")
        with BusyNotice(
            self, f"Finding the {self.num_units_to_relabel} least confident units",
            f"Ranking the units of {len(targets)} analyzers by the confidence of {self.selection_description()}. "
            "Analyzers without saved predictions are labelled by the model(s) first, which takes longer the first time.",
        ):
            num_picked, skipped = find_least_confident_units(targets, mode, model_folders, selection_key, self.num_units_to_relabel)

        for target_index, target in enumerate(targets):
            print(f"    {target['path']}: {num_picked[target_index]} units")
        for target, error in skipped:
            print(f"    Skipped {target['path']}: {error}")
        print()

        self.make_relabel_button_list()

        if len(skipped) == len(targets):
            self.show_problem(
                "Could not rank the units",
                "The chosen model(s) could not label any of the analyzers:\n\n"
                + "\n".join(f"• {Path(str(target['path'])).name}: {error}" for target, error in skipped),
            )
        elif sum(num_picked.values()) == 0:
            self.show_problem(
                "No units left to relabel",
                "Every unit of your analyzers already has a label (curated, imported or relabelled), so there is "
                "nothing new to ask about. Add new analyzers in section 3.3 to continue.",
            )

    def add_from_hfh(self):

        existing_repo_ids = []
        for model_folder, hfh_or_local in self.project.model_paths:
            hfh_path = Path(model_folder) / 'hfh_path.txt'
            if hfh_or_local == "hfh" and hfh_path.is_file():
                existing_repo_ids.append(hfh_path.read_text().strip())

        dialog = HFHModelsDialog(existing_repo_ids, parent=self)
        if not dialog.exec():
            return

        repo_ids = dialog.get_repo_ids()
        if len(repo_ids) == 0:
            print("No models selected.")
            return

        added_repo_ids = []
        with BusyNotice(self, f"Downloading {len(repo_ids)} model{'s' if len(repo_ids) > 1 else ''} from the Hugging Face Hub", "") as notice:
            for repo_index, repo_id in enumerate(repo_ids):
                notice.update(f"{repo_index + 1} of {len(repo_ids)}: {repo_id}")
                try:
                    if self.download_hfh_model(repo_id):
                        added_repo_ids.append(repo_id)
                except Exception as e:
                    print(f"Could not download model from {repo_id}: {e}\n")

        self.make_model_list()
        print(f"Added {len(added_repo_ids)}/{len(repo_ids)} models to project: {added_repo_ids}\n")

    def download_hfh_model(self, url):
        """Downloads one model from the Hugging Face Hub into the project. Returns True if it was added."""

        if not repo_exists(url):
            print(f"Repo {url} does not exist.")
            return False

        model_directory = self.project.folder_name / "models" / url.split('/')[-1]
        if model_directory.is_dir():
            print(f"Skipping {url}: a model folder called {model_directory.name} already exists in the project.")
            return False

        from huggingface_hub import snapshot_download

        print(f"Downloading model from HuggingFaceHub repo {url}.\n")

        try:
            snapshot_download(url, local_dir=model_directory)
        except Exception:
            # don't leave a half-downloaded model in the project
            shutil.rmtree(model_directory, ignore_errors=True)
            raise

        with open(model_directory / 'hfh_path.txt', 'w') as output:
            output.write(url)

        model_info_path = model_directory / "model_info.json"
        with open(model_info_path, 'r') as f:
            model_info = json.load(f)

        updated_model_info = {}
        translation_dict = {
            'MUA': ['mua'],
            'good': ['sua', 'SUA', 'neural'],
            'noise': [],
        }
        for label_index, label in model_info['label_conversion'].items():
            new_label = label
            for sigui_label, user_labels in translation_dict.items():
                if label in user_labels:
                    new_label = sigui_label

            updated_model_info[label_index] = new_label
        model_info['label_conversion'] = updated_model_info

        with open(model_info_path, 'w') as f:
            json.dump(model_info, f, indent=4)

        self.project.model_paths = self.project.model_paths + [(model_directory, "hfh")]

        print(f"Model at {url} successfully added to project.\n")
        return True

    def make_apply_code(self):

        problem = self.model_selector.describe_problem()
        if problem is not None:
            print(problem)
            return

        model_folders = [model_folder for model_folder, _ in self.model_selector.selected_models()]

        # the function names of the installed SpikeInterface version (auto_label_units was renamed)
        label_units = label_units_function().__name__
        import spikeinterface.curation as spikeinterface_curation
        has_cascade_function = hasattr(spikeinterface_curation, "unitrefine_label_units")

        code_text = "\n"
        code_text += "import spikeinterface.full as si\n"
        code_text += "import spikeinterface.curation as sc\n\n"
        code_text += "# point this path to the analyzer you want to apply the model to\n"
        code_text += "path_to_analyzer = 'path/to/analyzer'\n"
        code_text += "analyzer_to_curate = si.load_sorting_analyzer(path_to_analyzer)\n\n"

        if self.model_selector.mode() == "cascade" and has_cascade_function:
            code_text += "# Cascade, as in the UnitRefine paper: noise vs neural, then SUA vs MUA on the neural units\n"
            code_text += (f"labels = sc.unitrefine_label_units(\n\tsorting_analyzer = analyzer_to_curate,\n"
                          f"\tnoise_neural_classifier = r'{model_folders[0]}',\n\tsua_mua_classifier = r'{model_folders[1]}',\n)\n\n")
        elif self.model_selector.mode() == "cascade":
            code_text += "# Cascade: layer 1 separates noise from neural units, layer 2 splits the neural units into SUA and MUA\n"
            code_text += f"layer1 = sc.{label_units}(\n\tsorting_analyzer = analyzer_to_curate,\n\tmodel_folder = r'{model_folders[0]}',\n\ttrust_model = True,\n)\n"
            code_text += f"layer2 = sc.{label_units}(\n\tsorting_analyzer = analyzer_to_curate,\n\tmodel_folder = r'{model_folders[1]}',\n\ttrust_model = True,\n)\n\n"
            code_text += "is_noise = layer1['prediction'] == 'noise'\n"
            code_text += f"layer2_labels = layer2['prediction'].map({QUALITY_ALIASES})\n"
            code_text += "labels = layer2_labels.where(~is_noise, 'noise')\n"
            code_text += "# confidence of the cascade: p(layer 1) x p(layer 2) for neural units, p(layer 1) for noise units\n"
            code_text += "confidence = (layer1['probability'] * layer2['probability']).where(~is_noise, layer1['probability'])\n\n"
        else:
            for model_folder in model_folders:
                code_text += f"# labels will be a DataFrame of curated labels and their probability, determined by {model_folder.name}.\n"
                code_text += f"labels = sc.{label_units}(\n\tsorting_analyzer = analyzer_to_curate,\n\tmodel_folder = r'{model_folder}',\n\ttrust_model = True,\n)\n\n"

        code_text += "# Read more here: https://spikeinterface.readthedocs.io/en/stable/tutorials/curation/plot_1_automated_curation.html\n\n"

        print(code_text)

    def selectDirectoryDialog(self):
        file_dialog = QtWidgets.QFileDialog(self)
        file_dialog.setWindowTitle("Select Directory")
        file_dialog.setFileMode(QtWidgets.QFileDialog.FileMode.Directory)
        file_dialog.setViewMode(QtWidgets.QFileDialog.ViewMode.List)

        if file_dialog.exec():
            selected_directory = file_dialog.selectedFiles()[0]

            if check_folder_is_analyzer(selected_directory):

                self.project.add_analyzer(selected_directory)
                self.project.save()

                self.make_curation_button_list()
                self.make_model_list()
                self.make_relabel_button_list()

                print(f"Successfully added analyzer at {selected_directory} to project.")
            else:
                print(f"Selected directory {selected_directory} is not a SortingAnalyzer.")

    def selectModelDialog(self):
        file_dialog = QtWidgets.QFileDialog(self)
        file_dialog.setWindowTitle("Select Directory")
        file_dialog.setFileMode(QtWidgets.QFileDialog.FileMode.Directory)
        file_dialog.setViewMode(QtWidgets.QFileDialog.ViewMode.List)

        if file_dialog.exec():
            selected_directory = file_dialog.selectedFiles()[0]

            if is_a_model(selected_directory):
                self.project.model_paths = self.project.model_paths + [(Path(selected_directory), "local")]
                self.make_model_list()
                print(f"Model at {selected_directory} added to project.")
            else:
                print(f"{selected_directory} is not a UnitRefine model folder.")
            
    def make_curation_button_list(self):

        clear_layout(self.analyzer_rows_layout)

        if len(self.project.analyzers) == 0:
            self.analyzer_rows_layout.addWidget(list_message("No analyzers yet: add a Sorting Analyzer folder or a Phy output to start."))

        for row_index, (analyzer_index, analyzer) in enumerate(self.project.analyzers.items()):

            selected_directory = analyzer['path']

            curate_button = theme.make_sigui(QtWidgets.QPushButton("Curate"))
            curate_button.setToolTip(f"Label the units of {selected_directory} in the SpikeInterface-GUI")
            curate_button.clicked.connect(
                lambda _checked=False, directory=selected_directory, index=analyzer_index: self.show_curation_window(directory, index))

            or_text = muted_label("or", word_wrap=False)

            curation_output_folder = Path(self.project.folder_name) / Path(f"analyzers/analyzer_{analyzer_index}")
            curation_output_folder.mkdir(exist_ok=True)

            imported_source_path = curation_output_folder / "imported_labels_source.txt"
            if imported_source_path.is_file():
                import_labels_button = QtWidgets.QPushButton("Imported ✓")
                import_labels_button.setToolTip(f"Labels imported from {imported_source_path.read_text()}. Click to import again.")
            else:
                import_labels_button = QtWidgets.QPushButton("Import labels")
                import_labels_button.setToolTip(
                    "Import labels from Phy (cluster_group.tsv, cluster_info.tsv), SortingView or SpikeInterface "
                    "curation (.json), or a .csv/.tsv with unit_id and quality columns"
                )
            import_labels_button.clicked.connect(lambda _checked=False, index=analyzer_index: self.import_labels_dialog(index))

            if (curation_output_folder / "all_metrics.csv").is_file() and (curation_output_folder / "labels.csv").is_file():
                num_labelled = len(pd.read_csv(curation_output_folder / "labels.csv"))
                num_units = len(pd.read_csv(curation_output_folder / "all_metrics.csv"))
                count_chip = chip(f"{num_labelled}/{num_units} labelled", done=num_labelled == num_units)
            else:
                count_chip = chip("not labelled")
            count_chip.setMinimumWidth(110)

            remove_button = QtWidgets.QToolButton()
            remove_button.setIcon(self.style().standardIcon(QtWidgets.QStyle.SP_TrashIcon))
            remove_button.setToolTip(f"Remove {selected_directory} from the project")
            remove_button.clicked.connect(lambda _checked=False, index=analyzer_index: self.remove_analyzer(index))

            self.analyzer_rows_layout.addWidget(PathRow(
                str(selected_directory), [curate_button, or_text, import_labels_button, count_chip, remove_button], first=row_index == 0))

        self.update_summaries()
        self.schedule_figure_refresh()

    def ask_label_file(self):
        """Asks for a label file; returns "" if cancelled."""
        label_file, _ = QtWidgets.QFileDialog.getOpenFileName(
            self,
            "Select label file",
            "",
            "All supported label files (*.tsv *.csv *.json);;"
            "Phy (cluster_group.tsv cluster_info.tsv cluster_KSLabel.tsv);;"
            "SortingView or SpikeInterface curation (*.json);;"
            "Table with unit_id and quality columns (*.csv *.tsv);;"
            "All files (*)",
        )
        return label_file

    def import_labels_dialog(self, analyzer_index):

        label_file = self.ask_label_file()
        if label_file == "":
            return
        self.import_label_file(analyzer_index, label_file)

    def import_label_file(self, analyzer_index, label_file):
        """Imports the labels in `label_file` for an analyzer in the project. Returns True if it worked."""

        analyzer_dict = self.project.analyzers[analyzer_index]
        save_folder = Path(self.project.folder_name) / Path(f"analyzers/analyzer_{analyzer_index}")

        labels_path = save_folder / "labels.csv"
        existing_labels = pd.read_csv(labels_path) if labels_path.is_file() else None

        print(f"\nImporting labels from {label_file} for analyzer at {analyzer_dict['path']}...")
        try:
            with BusyNotice(self, "Importing labels", f"Reading {Path(label_file).name} and the metrics of the analyzer's units."):
                labels_df = import_labels(label_file, analyzer_dict['path'], save_folder, existing_labels=existing_labels)
        except Exception as e:
            print(f"Could not import labels: {e}")
            QtWidgets.QMessageBox.warning(self, "Could not import labels", str(e))
            return False

        # keep the in-memory project in sync, so a later project.save() does not overwrite the imported labels
        analyzer_dict['labels'] = labels_df
        (save_folder / "imported_labels_source.txt").write_text(str(label_file))

        self.make_curation_button_list()
        return True

    def show_unwritable_file(self, path, message):
        if str(path) in self.reported_unwritable_files:
            return
        self.reported_unwritable_files.add(str(path))
        QtWidgets.QMessageBox.warning(self, "Could not save a file", message)

    def confirm(self, title, text):
        """Asks Yes or No, with No as the default; returns True for Yes."""
        answer = QtWidgets.QMessageBox.question(
            self, title, text, QtWidgets.QMessageBox.Yes | QtWidgets.QMessageBox.No, QtWidgets.QMessageBox.No)
        return answer == QtWidgets.QMessageBox.Yes

    def remove_analyzer(self, analyzer_index):

        analyzer = self.project.analyzers[analyzer_index]
        analyzer_folder = Path(self.project.folder_name) / Path(analyzer['analyzer_in_project'])

        # the folder holds the labels curated or imported for this analyzer, which can't be brought back
        labels_path = analyzer_folder / "labels.csv"
        num_labels = len(pd.read_csv(labels_path)) if labels_path.is_file() else 0
        losing = (f"its {num_labels} labels, which can't be brought back" if num_labels > 0
                  else "its files in the project")
        if not self.confirm(
            "Remove analyzer?",
            f"Remove {analyzer['path']} from the project and delete {losing}?\n\n"
            "The recording itself is not deleted.",
        ):
            return

        shutil.rmtree(str(analyzer_folder))
        if analyzer_folder.is_dir():
            os.rmdir(str(analyzer_folder))

        print(f"Removing analyzer at {analyzer_folder} from project.")

        self.project.analyzers = {key: value for key, value in self.project.analyzers.items() if key != analyzer_index}

        self.make_curation_button_list()
        self.make_validate_button_list()
        self.make_relabel_button_list()

    def make_relabel_button_list(self):

        clear_layout(self.relabel_rows_layout)

        selection_key = self.model_selector.selection_key()
        targets = self.relabel_targets()

        if len(targets) == 0:
            self.relabel_rows_layout.addWidget(list_message(
                "No analyzers yet: add them in section 2, or apply the models to new analyzers in section 3.3."))

        total_relabelled, total_picked = 0, 0
        for target_index, target in enumerate(targets):

            # same groups as in section 3.3
            is_first_new = target['analyzer_index'] is None and (target_index == 0 or targets[target_index - 1]['analyzer_index'] is not None)
            if target_index == 0 and target['analyzer_index'] is not None:
                self.relabel_rows_layout.addWidget(group_label("Analyzers used during training the models:"))
            elif is_first_new:
                self.relabel_rows_layout.addWidget(group_label("New analyzers the models were applied to:"))

            relabel_button = theme.make_sigui(QtWidgets.QPushButton("Relabel"))
            # lambda drops the `checked` argument which clicked passes
            relabel_button.clicked.connect(lambda _checked=False, target=target: self.show_validate_window(
                target['analyzer'], target['analyzer_index'], True, save_folder=target['save_folder'], file_prefix=target['file_prefix']))

            status_chip = chip("not searched yet")
            relabel_button.setToolTip("Click Find least confident units first")
            if selection_key is not None and units_to_relabel_path(target, selection_key).is_file():
                units_to_relabel = pd.read_csv(units_to_relabel_path(target, selection_key))
                num_relabelled = 0
                if relabelled_units_path(target, selection_key).is_file():
                    relabelled = set(pd.read_csv(relabelled_units_path(target, selection_key)).dropna(subset=['quality'])['unit_id'].astype(str))
                    num_relabelled = int(units_to_relabel['unit_id'].astype(str).isin(relabelled).sum())
                total_relabelled += num_relabelled
                total_picked += len(units_to_relabel)
                if len(units_to_relabel) == 0:
                    status_chip = chip("no units picked")
                    relabel_button.setEnabled(False)
                    relabel_button.setToolTip("None of the least confident units are in this analyzer")
                else:
                    status_chip = chip(f"{num_relabelled}/{len(units_to_relabel)} relabelled", done=num_relabelled == len(units_to_relabel))
                    relabel_button.setToolTip("Open the least confident units of this analyzer in the SpikeInterface-GUI")
            status_chip.setMinimumWidth(120)

            self.relabel_rows_layout.addWidget(PathRow(str(target['path']), [status_chip, relabel_button], first=False))

        self.relabel_section.set_summary(f"{total_relabelled}/{total_picked} least confident units relabelled" if total_picked > 0 else "")
        self.make_retraining_history()
        self.schedule_figure_refresh()

    def make_retraining_history(self):
        """The retraining rounds of the chosen models: what each added, from where, and the change in accuracy."""
        names = self.model_selector.selected_model_names()
        lineages = dict(self.model_lineage(name) for name in names)
        num_rounds = self.history_table.set_lineages(lineages)
        self.history_table.setVisible(num_rounds > 0)
        if len(names) == 0:
            self.history_message.setText("Choose a model in 3.2 to see its retraining rounds.")
        elif num_rounds == 0:
            self.history_message.setText("The chosen model(s) have not been retrained yet: each round you retrain appears here.")
        else:
            self.history_message.setText(
                "Each round: the units you added and where they come from (<b>Details</b> lists every unit with its old "
                "and new label), and the balanced accuracy on held-out units of the same classifier trained without, "
                "then with them.")

    def make_validate_button_list(self):

        clear_layout(self.validate_rows_layout)

        # the new analyzers: predictions on recordings the models were not trained on
        new_list, new_rows_layout = list_box()
        applied_analyzers = self.project.config.get("applied_analyzers", [])
        if len(applied_analyzers) == 0:
            new_rows_layout.addWidget(list_message("No new analyzers yet: add them with the buttons above."))

        for applied_index, applied in enumerate(applied_analyzers):
            file_prefix = f"{applied['output_name']}__"
            labels_path = Path(applied['output_folder']) / f"{file_prefix}labels.csv"

            if labels_path.is_file():
                import_labels_button = QtWidgets.QPushButton("Imported ✓")
                source_path = Path(applied['output_folder']) / f"{file_prefix}imported_labels_source.txt"
                source = source_path.read_text() if source_path.is_file() else str(labels_path)
                import_labels_button.setToolTip(f"Labels imported from {source}. Click to import again.")
                labels_chip = chip(f"{len(pd.read_csv(labels_path))} labelled", done=True)
                labels_chip.setMinimumWidth(90)
            else:
                import_labels_button = QtWidgets.QPushButton("Import labels")
                import_labels_button.setToolTip("Optional: your labels for this analyzer, to compare them with the predictions")
                # the Import labels button already says there are none
                labels_chip = None
            import_labels_button.clicked.connect(lambda _checked=False, applied=applied: self.import_labels_for_new_analyzer(applied))

            inspect_button = theme.make_sigui(QtWidgets.QPushButton("Inspect"))
            inspect_button.setToolTip(f"Open in the SpikeInterface-GUI with the models' labels and confidence. "
                                      f"Predictions are saved in {applied['output_folder']}")
            inspect_button.clicked.connect(lambda _checked=False, applied=applied: self.show_applied_analyzer_window(applied))

            remove_button = QtWidgets.QToolButton()
            remove_button.setIcon(self.style().standardIcon(QtWidgets.QStyle.SP_TrashIcon))
            remove_button.setToolTip("Remove from this list (saved csv files are kept)")
            remove_button.clicked.connect(lambda _checked=False, index=applied_index: self.remove_applied_analyzer(index))
            new_rows_layout.addWidget(PathRow(
                applied['path'], [widget for widget in [import_labels_button, labels_chip, inspect_button, remove_button] if widget is not None], first=applied_index == 0))
        self.validate_rows_layout.addWidget(new_list)
        self.validate_rows_layout.addWidget(muted_label(
            "<b>Import labels</b> (optional) to see how a model performs on your recordings. <b>Import labels</b> reads "
            "Phy (cluster_group.tsv, cluster_info.tsv), SortingView (.json), SpikeInterface curation (.json), or a "
            ".csv/.tsv with <i>unit_id</i> and <i>quality</i> columns. Labels good, MUA and noise are used; others "
            "(e.g. unsorted) are skipped."))

        # the training analyzers are hidden at first: predictions on training data look better than they are
        toggle_button = theme.make_flat(QtWidgets.QPushButton(
            ("▾ Hide" if self.show_training_analyzers else "▸ Show") + " the analyzers used for training"))
        toggle_button.setToolTip("Inspecting them shows what the model learned, but not how well it does: "
                                 "predictions on training data look better than they are")
        toggle_button.clicked.connect(self.toggle_training_analyzers)
        toggle_layout = QtWidgets.QHBoxLayout()
        toggle_layout.addWidget(toggle_button)
        toggle_layout.addStretch()
        self.validate_rows_layout.addLayout(toggle_layout)

        if self.show_training_analyzers:
            training_list, training_rows_layout = list_box()
            if len(self.project.analyzers) == 0:
                training_rows_layout.addWidget(list_message("None yet: add analyzers in section 2."))
            for analyzer_index, analyzer in enumerate(self.project.analyzers.values()):
                inspect_button = theme.make_sigui(QtWidgets.QPushButton("Inspect"))
                inspect_button.setToolTip("Open the analyzer in the SpikeInterface-GUI with the models' labels and confidence")
                inspect_button.clicked.connect(lambda _checked=False, analyzer=analyzer, analyzer_index=analyzer_index: self.show_validate_window(analyzer, analyzer_index, False))
                training_rows_layout.addWidget(PathRow(str(analyzer['path']), [inspect_button], first=analyzer_index == 0))
            self.validate_rows_layout.addWidget(training_list)

        self.update_summaries()
        self.schedule_figure_refresh()

    def selection_description(self):
        names = self.model_selector.selected_model_names()
        if self.model_selector.mode() == "cascade":
            return f"cascade of {names[0]} (layer 1) and {names[1]} (layer 2)"
        return ", ".join(names)

    def show_problem(self, title, message):
        print(message)
        QtWidgets.QMessageBox.information(self, title, message)

    def add_new_analyzers_from_folder(self):
        """Adds one new analyzer, or all analyzers inside a folder, and applies the chosen models to them."""

        problem = self.model_selector.describe_problem()
        if problem is not None:
            self.show_problem("Choose a model first", problem)
            return

        folder = QtWidgets.QFileDialog.getExistingDirectory(self, "Select a Sorting Analyzer folder, or a folder containing several")
        if folder == "":
            return
        analyzer_paths = [str(path) for path in find_analyzers(folder)]
        if len(analyzer_paths) == 0:
            self.show_problem("No Sorting Analyzer found", f"No Sorting Analyzer was found in {folder} (searched 3 folder levels down).")
            return
        self.apply_to_new_analyzers(analyzer_paths)

    def add_new_analyzer_from_s3(self):

        problem = self.model_selector.describe_problem()
        if problem is not None:
            self.show_problem("Choose a model first", problem)
            return

        dialog = UrlInputDialog()
        if not dialog.exec():
            return
        url = dialog.get_url()
        if not is_path_remote(url):
            self.show_problem("Not an s3 path", f"{url} is not a valid analyzer path.")
            return
        self.apply_to_new_analyzers([url])

    def apply_to_new_analyzers(self, analyzer_paths):
        """
        Applies the chosen models to new analyzers, saving their predictions in the project's predictions folder,
        and adds them to the list in 3.3. Analyzers already in the list are skipped.
        """
        mode = self.model_selector.mode()
        model_folders = [model_folder for model_folder, _ in self.model_selector.selected_models()]
        selection_key = self.model_selector.selection_key()
        output_folder = Path(self.project.folder_name) / "predictions"

        applied_analyzers = self.project.config.get("applied_analyzers", [])
        listed = {normalise_path(applied['path']) for applied in applied_analyzers}
        already_listed = [path for path in analyzer_paths if normalise_path(path) in listed]
        analyzer_paths = [path for path in analyzer_paths if normalise_path(path) not in listed]
        if len(analyzer_paths) == 0:
            self.show_problem("Already added", "These analyzers are already in the list.")
            return

        # file names must not clash with the analyzers already in the list, which keep theirs
        used_names = {applied['output_name'] for applied in applied_analyzers}
        candidate_names = analyzer_output_names([applied['path'] for applied in applied_analyzers] + analyzer_paths)[len(applied_analyzers):]
        output_names = []
        for name in candidate_names:
            unique_name, number = name, 2
            while unique_name in used_names:
                unique_name, number = f"{name}_{number}", number + 1
            used_names.add(unique_name)
            output_names.append(unique_name)

        with BusyNotice(
            self, f"Applying {self.selection_description()} to {len(analyzer_paths)} analyzer{'s' if len(analyzer_paths) > 1 else ''}",
            "The predictions are saved, so they are only computed once.",
        ):
            results = apply_to_analyzers(analyzer_paths, mode, model_folders, selection_key, output_folder, output_names=output_names)

        # remember the analyzers, so they can be inspected later
        for result in results:
            if "error" not in result:
                applied_analyzers.append(dict(path=result['path'], output_folder=str(output_folder), output_name=result['output_name']))
        self.project.config["applied_analyzers"] = applied_analyzers
        self.project.save_config()

        failed = [result for result in results if "error" in result]
        print(f"\nApplied {self.selection_description()} to {len(results) - len(failed)}/{len(results)} analyzers. Predictions saved in {output_folder}.\n")
        if len(failed) > 0 or len(already_listed) > 0:
            lines = [f"• {Path(str(result['path'])).name}: {result['error']}" for result in failed]
            lines += [f"• {Path(str(path)).name}: already in the list" for path in already_listed]
            QtWidgets.QMessageBox.warning(self, "Some analyzers were not added", "\n\n".join(lines))

        self.make_validate_button_list()
        self.make_relabel_button_list()

    def import_labels_for_new_analyzer(self, applied):
        """Labels of a new analyzer, which the models are compared with (they are never trained on them)."""

        label_file = self.ask_label_file()
        if label_file == "":
            return
        self.import_label_file_for_new_analyzer(applied, label_file)

    def import_label_file_for_new_analyzer(self, applied, label_file):
        """Imports the labels in `label_file` for a new analyzer (section 3.3). Returns True if it worked."""
        save_folder = Path(applied['output_folder'])
        file_prefix = f"{applied['output_name']}__"
        labels_path = save_folder / f"{file_prefix}labels.csv"
        existing_labels = pd.read_csv(labels_path) if labels_path.is_file() else None

        print(f"\nImporting labels from {label_file} for analyzer at {applied['path']}...")
        try:
            with BusyNotice(self, "Importing labels", f"Reading {Path(label_file).name} and the metrics of the analyzer's units."):
                import_labels(label_file, applied['path'], save_folder, existing_labels=existing_labels, file_prefix=file_prefix)
        except Exception as e:
            print(f"Could not import labels: {e}")
            QtWidgets.QMessageBox.warning(self, "Could not import labels", str(e))
            return False
        (save_folder / f"{file_prefix}imported_labels_source.txt").write_text(str(label_file))

        self.make_validate_button_list()
        return True

    # ----------------------------------------------------------------------------------------------
    # Phy output (+ Phy output, in sections 2 and 3.3)
    # ----------------------------------------------------------------------------------------------

    def add_phy_output(self, for_training):
        """
        Adds a Phy folder: makes its analyzer (see phy_import.py) in a separate process, then adds it to section 2
        (`for_training`) or applies the chosen models to it (section 3.3), with Phy's labels imported.
        """
        if not for_training:
            problem = self.model_selector.describe_problem()
            if problem is not None:
                self.show_problem("Choose a model first", problem)
                return
        if self.phy_process is not None:
            self.show_problem("Already making an analyzer", "Wait until the Phy folder being added is finished.")
            return

        phy_folder = QtWidgets.QFileDialog.getExistingDirectory(self, "Select a Phy folder (the folder with params.py)")
        if phy_folder == "":
            return
        phy_folder = Path(phy_folder)
        try:
            kind = phy_folder_kind(phy_folder)
        except ValueError as e:
            self.show_problem("Not a Phy folder", str(e))
            return

        output_folder = self.phy_output_folder(phy_folder)
        if output_folder is None:
            return
        labels_path = Path(self.project.folder_name) / "phy_labels" / f"{output_folder.name}_phy_labels.csv"
        arguments = [phy_folder, "--output", output_folder, "--labels", labels_path]

        if kind == "exported":
            analyzer_path = self.analyzer_exported_to(phy_folder)
            if analyzer_path is None:
                return
            arguments += ["--analyzer", analyzer_path]
            steps, what = EXPORTED_STEPS, f"Applying the curation in {phy_folder.name} to {Path(str(analyzer_path)).name}"
        else:
            recording_path = recording_file(phy_folder)
            if recording_path is None:
                recording_path, _ = QtWidgets.QFileDialog.getOpenFileName(
                    self, "The recording Kilosort sorted was not found: select it (e.g. the .ap.bin file)",
                    str(phy_folder), "Recordings (*.bin *.dat *.cbin *.raw);;All files (*)")
                if recording_path == "":
                    return
                arguments += ["--recording", recording_path]
            if recording_needs_gain(recording_path):
                gain, accepted = QtWidgets.QInputDialog.getDouble(
                    self, "Gain of the recording",
                    f"{Path(recording_path).name} is a plain binary file, which doesn't say how to convert its values "
                    f"to µV.\nGive its gain in µV per bit ({GAIN_HINT}):", 1.0, 0.0001, 1000.0, 4)
                if not accepted:
                    return
                arguments += ["--gain_to_uv", gain]
            steps, what = KILOSORT_STEPS, f"Making an analyzer from {phy_folder.name}"

        self.run_phy_import(arguments, steps, what, output_folder, for_training)

    def phy_output_folder(self, phy_folder):
        """
        Where the analyzer made from a Phy folder is saved: next to the Phy folder, or elsewhere if that can't be
        written. If an analyzer made from this Phy folder exists, asks whether to use it or make it again.
        Returns None if cancelled.
        """
        output_folder = default_output_folder(phy_folder)
        made_from = made_from_phy(output_folder)
        if made_from is not None and normalise_path(made_from["phy_folder"]) == normalise_path(phy_folder):
            box = QtWidgets.QMessageBox(self)
            box.setWindowTitle("Analyzer already made")
            box.setText(f"An analyzer was made from {phy_folder.name} on {made_from['date'].replace('T', ' ')}:\n{output_folder}\n\n"
                        "Make it again if you curated more in Phy since then.")
            use_button = box.addButton("Use it", QtWidgets.QMessageBox.AcceptRole)
            box.addButton("Make it again", QtWidgets.QMessageBox.DestructiveRole)
            cancel_button = box.addButton(QtWidgets.QMessageBox.Cancel)
            box.exec()
            if box.clickedButton() is cancel_button:
                return None
            if box.clickedButton() is use_button:
                self.phy_reuse = True
            return output_folder
        number = 2
        while output_folder.exists():
            # e.g. a folder of the same name, not made from this Phy folder
            output_folder = default_output_folder(phy_folder).with_name(f"{default_output_folder(phy_folder).name}_{number}")
            number += 1
        if not os.access(output_folder.parent, os.W_OK):
            chosen = QtWidgets.QFileDialog.getExistingDirectory(
                self, f"{output_folder.parent} can't be written: choose where to save the analyzer")
            if chosen == "":
                return None
            output_folder = Path(chosen) / output_folder.name
        return output_folder

    def analyzer_exported_to(self, phy_folder):
        """The analyzer a Phy folder was exported from: found among the analyzers in the project, or asked for."""
        candidates = [analyzer['path'] for analyzer in self.project.analyzers.values()]
        candidates += [applied['path'] for applied in self.project.config.get("applied_analyzers", [])]
        with BusyNotice(self, "Finding the analyzer", f"Looking for the analyzer {phy_folder.name} was exported from."):
            for path in candidates:
                if is_path_remote(str(path)):
                    continue
                try:
                    sorting = load_sorting_analyzer(str(path), load_extensions=False).sorting
                except Exception:
                    continue
                if matches_phy_folder(sorting, phy_folder):
                    print(f"{phy_folder.name} was exported from {path}.")
                    return path
        folder = QtWidgets.QFileDialog.getExistingDirectory(
            self, f"{phy_folder.name} was exported from a SpikeInterface analyzer: select that analyzer")
        if folder == "":
            return None
        if not check_folder_is_analyzer(folder):
            self.show_problem("Not a Sorting Analyzer", f"{folder} is not a Sorting Analyzer.")
            return None
        return folder

    def run_phy_import(self, arguments, steps, what, output_folder, for_training):
        """Runs phy_import in its own process, with a progress dialog, then adds the analyzer (see on_phy_result)."""
        if self.phy_reuse:
            # an analyzer made earlier from this Phy folder: only the labels are read again (fast)
            self.phy_reuse = False
            arguments = arguments + ["--reuse"]
        process = QProcess(self)
        process.setProcessChannelMode(QProcess.MergedChannels)
        environment = QProcessEnvironment.systemEnvironment()
        environment.insert("PYTHONUNBUFFERED", "1")
        environment.insert("PYTHONIOENCODING", "utf-8")
        process.setProcessEnvironment(environment)
        self.phy_process = process
        self.phy_output_buffer = ""
        self.phy_result = None
        self.phy_cancelled = False

        timing_key = normalise_path(arguments[0])
        previous_seconds = self.project.config.get("phy_import_seconds", {}).get(timing_key)
        self.phy_dialog = OpeningDialog(
            what, steps, previous_seconds, parent=self, window_title="Adding Phy output",
            note=(f"Your Phy folder is not changed. The analyzer is saved in<br><i>{output_folder}</i><br>"
                  "The terminal shows the details."),
            slow_message=("This takes a few minutes for a long recording, most of it extracting the waveforms. "
                          "Nothing has crashed, please don't close anything."),
        )
        self.phy_dialog.cancelled.connect(self.cancel_phy_import)
        self.phy_dialog.show()
        self.content_widget.setEnabled(False)

        def finished(exit_code, exit_status):
            self.on_phy_finished(arguments, steps, what, output_folder, for_training, timing_key)

        process.readyReadStandardOutput.connect(self.on_phy_output)
        process.finished.connect(finished)
        print(f"\n{what}...")
        process.start(sys.executable, ["-m", "unitrefine.phy_import", *[str(argument) for argument in arguments + self.phy_import_options]])

    def on_phy_output(self):
        output = bytes(self.phy_process.readAllStandardOutput()).decode("utf-8", errors="replace")
        self.phy_output_buffer += output.replace("\r\n", "\n")
        *lines, self.phy_output_buffer = self.phy_output_buffer.split("\n")
        for line in lines:
            if line.startswith(PHY_STEP_MARKER):
                self.phy_dialog.set_step(line[len(PHY_STEP_MARKER):].strip())
            elif line.startswith(PHY_RESULT_MARKER):
                self.phy_result = json.loads(line[len(PHY_RESULT_MARKER):])
            else:
                sys.stdout.write(line + "\n")
        sys.stdout.flush()

    def cancel_phy_import(self):
        if self.phy_process is not None:
            print("Cancelled adding the Phy folder.")
            self.phy_cancelled = True
            self.phy_process.kill()

    def on_phy_finished(self, arguments, steps, what, output_folder, for_training, timing_key):
        self.phy_dialog.done_opening()
        elapsed = self.phy_dialog.elapsed_seconds()
        self.phy_process = None
        self.content_widget.setEnabled(True)
        result = self.phy_result or dict(status="error", message="It stopped without a result. The terminal shows the messages.")
        if self.phy_cancelled:
            return

        if result["status"] == "hard_merge_required":
            if self.confirm("Merges need the raw recording", f"{result['message']}\n\nDo these merges? It can take a long time."):
                self.run_phy_import(arguments + ["--allow_hard_merge"], steps, what, output_folder, for_training)
            return
        if result["status"] != "done":
            QtWidgets.QMessageBox.warning(self, "Could not add the Phy folder", result["message"])
            return

        self.project.config.setdefault("phy_import_seconds", {})[timing_key] = round(elapsed, 1)
        self.project.save_config()
        self.add_phy_analyzer(result, for_training)

    def add_phy_analyzer(self, result, for_training):
        """Adds the analyzer made from a Phy folder to section 2 or 3.3, and imports Phy's labels for it."""
        analyzer_path = result["analyzer"]
        has_labels = result["num_labels"] > 0
        if for_training:
            existing = [index for index, analyzer in self.project.analyzers.items()
                        if normalise_path(analyzer['path']) == normalise_path(analyzer_path)]
            if len(existing) > 0:
                analyzer_index = existing[0]
            else:
                self.project.add_analyzer(analyzer_path)
                self.project.save()
                analyzer_index = max(self.project.analyzers)
            if has_labels:
                self.import_label_file(analyzer_index, result["labels"])
            self.make_curation_button_list()
            self.make_model_list()
            self.make_relabel_button_list()
            where = "section 2"
        else:
            listed = [applied for applied in self.project.config.get("applied_analyzers", [])
                      if normalise_path(applied['path']) == normalise_path(analyzer_path)]
            if len(listed) == 0:
                self.apply_to_new_analyzers([analyzer_path])
                listed = [applied for applied in self.project.config.get("applied_analyzers", [])
                          if normalise_path(applied['path']) == normalise_path(analyzer_path)]
            if len(listed) == 0:
                return
            if has_labels:
                self.import_label_file_for_new_analyzer(listed[0], result["labels"])
            where = "section 3.3"

        labels_text = (f"{result['num_labels']} units have a label from Phy, which were imported."
                       if has_labels else "No units were labelled in Phy.")
        print(f"Added {analyzer_path} to {where}. {labels_text}")
        QtWidgets.QMessageBox.information(self, "Phy folder added", f"The analyzer is added to {where}:\n{analyzer_path}\n\n{labels_text}")

    def toggle_training_analyzers(self):
        self.show_training_analyzers = not self.show_training_analyzers
        self.make_validate_button_list()

    def remove_applied_analyzer(self, applied_index):
        applied_analyzers = self.project.config.get("applied_analyzers", [])
        self.project.config["applied_analyzers"] = [applied for index, applied in enumerate(applied_analyzers) if index != applied_index]
        self.project.save_config()
        self.make_validate_button_list()

    def show_applied_analyzer_window(self, applied):
        analyzer = {'path': applied['path']}
        self.show_validate_window(analyzer, None, False, save_folder=applied['output_folder'], file_prefix=f"{applied['output_name']}__")


    def retrain_model(self):
        """Retrains every selected model (both layers for a cascade) with the units relabelled for this selection."""

        problem = self.model_selector.describe_problem()
        if problem is not None:
            print(problem)
            return

        names_to_retrain = self.models_to_retrain()
        selected_models = [model for model in self.model_selector.selected_models() if model[0].name in names_to_retrain]
        selection_key = self.model_selector.selection_key()

        if len(selected_models) == 1:
            retrained_names = [self.retrainedModelNameForm.text().strip()]
        else:
            retrained_names = self.retrained_names()

        results = []
        with BusyNotice(self, "Retraining", "") as notice:
            for (model_folder, _), retrained_name in zip(selected_models, retrained_names):
                notice.update(f"Retraining {model_folder.name} as {retrained_name}, with the units you relabelled.")
                print(f"\nRetraining {model_folder.name} as {retrained_name}...")
                results.append((model_folder.name, self.retrain_one_model(model_folder, retrained_name, selection_key)))

        self.make_model_list()
        self.show_retrain_summary(results)

    def show_retrain_summary(self, results):
        """Tells the user what retraining did, and what they can do next."""

        lines = []
        for model_name, result in results:
            if not result.get("retrained", True):
                reasons = "".join(f"<br>• {reason}" for reason in result["reasons"])
                lines.append(f"<li><b>{model_name}</b>: not retrained, none of your relabelled units could be added:"
                             f"<span style='color:{theme.DANGER}'>{reasons}</span></li>")
            else:
                recordings = ", ".join(f"{Path(str(path)).parent.name}/{Path(str(path)).name} ({len(unit_ids)})"
                                       for path, unit_ids in result['recordings'].items())
                change = result['held_out_balanced_accuracy'] - result['held_out_before']
                lines.append(
                    f"<li><b>{model_name}</b> → <b>{result['name']}</b>: learned from {result['num_units_added']} "
                    f"relabelled units ({recordings}). Held-out balanced accuracy {result['held_out_before']:.2f} → "
                    f"{result['held_out_balanced_accuracy']:.2f} ({change:+.2f}) on {result['num_held_out']} units.</li>"
                )
        retrained_any = any(result.get("retrained", True) for _, result in results)

        if retrained_any:
            next_steps = (
                "<p>The retrained model(s) are now selected in <b>3.2 Choose your models</b>. What you can do next:</p><ul>"
                "<li><b>See what changed</b>: the <i>Retraining history</i> in section 4 lists every round, and "
                "<b>Details</b> every unit added with its old and new label.</li>"
                "<li><b>Check whether it improved</b>: look at the <i>Accuracy over rounds</i> figure in section 4 "
                "(it is updated in the background, in a few seconds).</li>"
                "<li><b>Another round of active learning</b>: in section 4, click <b>Find least confident units</b>, "
                "relabel them, and retrain again.</li>"
                "<li><b>Done?</b> If the accuracy no longer improves, or the least confident units look right to you, "
                "the model is ready: use <b>+ Apply to new analyzers</b> (section 3.3) to label your recordings.</li></ul>"
            )
        elif all(result.get("no_relabels") for _, result in results):
            next_steps = (
                "<p>Relabel some units first: in section 4, click <b>Find least confident units</b>, then "
                "<b>Relabel</b>, and give units a label different from the model's or a new label.</p>"
            )
        else:
            next_steps = "<p>The terminal shows more details.</p>"

        message_box = QtWidgets.QMessageBox(self)
        message_box.setWindowTitle("Retraining finished" if retrained_any else "Nothing to retrain")
        message_box.setTextFormat(Qt.TextFormat.RichText)
        message_box.setText(f"<ul>{''.join(lines)}</ul>{next_steps}")
        message_box.exec()

    def retrain_one_model(self, model_folder, retrained_model_name, selection_key):
        from sklearn.model_selection import train_test_split
        from sklearn.metrics import balanced_accuracy_score

        model_folder = Path(model_folder)
        retrained_model_folder = model_folder.parent / retrained_model_name

        model, model_info = load_model(
            model_folder=model_folder, trust_model=True
        )
        label_conversion = model_info['label_conversion']
        invert_label_conversion = {val: int(key) for key, val in label_conversion.items()}
        model_metric_names = model.feature_names_in_
        model_classes = list(invert_label_conversion.keys())

        def _map_quality(q: str):
            # map GUI labels to model labels to handle cascade models, e.g. trained with "not-sua" or "neural"
            return map_quality_to_model_class(q, model_classes)

        
        # keep the imputer as fitted: changing keep_empty_features after fitting adds back metrics that
        # were empty during training, so the imputer output no longer matches what the scaler expects
        model_imputer = model['imputer']

        model_scaler = model['scaler']

        X_training_raw = pd.read_csv(model_folder / 'training_data.csv')
        X_training = X_training_raw.drop('unit_id', axis=1).values
        Y_training_raw = pd.read_csv(model_folder / 'labels.csv')
        Y_training = Y_training_raw.drop('unit_index', axis=1).values
        label_column = Y_training_raw.drop('unit_index', axis=1).columns[0]

        y_training = Y_training.ravel()

        # which analyzer and unit each training row comes from, so a relabelled unit the model was trained on gets
        # its label changed in its own row, instead of a second row with the other label. Not known for models
        # trained before this was saved, or downloaded ones: their relabelled units are added as new rows
        training_rows = read_training_units(model_folder, len(y_training))
        if training_rows is None:
            print(f"{model_folder.name} has no {TRAINING_UNITS_FILE}, so relabelled units it was trained on are added as new rows.")
            row_of_unit = None
        else:
            row_of_unit = {(analyzer, unit_id): row for row, (analyzer, unit_id) in
                           enumerate(zip(training_rows['analyzer'], training_rows['unit_id']))}

        # Split indices so we can keep unit_id aligned too
        idx = np.arange(len(y_training))
        train_idx, test_idx = train_test_split(
            idx,
            test_size=0.2,
            random_state=0,          # keep stable across retrains
            stratify=y_training if len(np.unique(y_training)) > 1 else None
        )

        # --- preprocess consistently using the already-fitted preprocessing,
        # as 32-bit floats, like SpikeInterface when it trains and applies models
        X_training_proc = model_scaler.transform(
            model_imputer.transform(as_model_input(pd.DataFrame(X_training, columns=model_metric_names)))
        )

        X_new = []
        X_just_values = []
        y_new = []
        # the analyzer and unit of each new row
        new_rows_units = []
        units_added = {}
        # new labels for rows of the training data, as {row: label code}
        changed_rows = {}
        # what the retraining log records about each added unit, and the relabelled units which were not added
        added_unit_details = []
        num_same_as_original = 0
        num_not_in_classes = 0

        # units already learned in an earlier round are not added again
        parent_trained_on, _ = read_trained_on(model_folder)
        num_already_learned = 0
        num_relabelled = 0
        analyzers_missing_metrics = {}

        # relabelled units from the training analyzers and from the new analyzers the models were applied to.
        # Your label is your label whichever model picked the unit, so the relabels made with any model selection
        # are used (when a unit was relabelled several times, the most recent label)
        for target in self.relabel_targets():

            save_path = Path(target['save_folder'])
            file_prefix = target['file_prefix']

            latest_relabels = {}
            relabel_files = sorted(save_path.glob(f"{file_prefix}relabelled_units_*.csv"), key=lambda path: path.stat().st_mtime)
            for relabel_file in relabel_files:
                picked_with = relabel_file.stem[len(f"{file_prefix}relabelled_units_"):]
                table = pd.read_csv(relabel_file)
                for quality, unit_id in zip(table['quality'], table['unit_id'].astype(str)):
                    if isinstance(quality, str):
                        latest_relabels[unit_id] = (quality, picked_with)
            if len(latest_relabels) == 0:
                continue
            num_relabelled += len(latest_relabels)
            relabelled_units = pd.DataFrame(
                [(quality, unit_id, picked_with) for unit_id, (quality, picked_with) in latest_relabels.items()],
                columns=['quality', 'unit_id', 'picked_with'],
            )

            all_metrics_path = save_path / f"{file_prefix}all_metrics.csv"
            all_metrics = pd.read_csv(all_metrics_path) if all_metrics_path.is_file() else None
            relabelled_unit_ids = set(relabelled_units['unit_id'].astype(str))
            # recompute the metrics if there is no file yet, or if it misses relabelled units (older versions
            # of the Relabel window saved the metrics of the units shown only)
            if all_metrics is None or not relabelled_unit_ids.issubset(set(all_metrics['unit_id'].astype(str))):
                analyzer = load_sorting_analyzer(target['path'], load_extensions=False)
                all_metrics = compute_all_metrics(analyzer)
                all_metrics.to_csv(all_metrics_path, index_label="unit_id")
                all_metrics = pd.read_csv(all_metrics_path)
            # the original labels: curated or imported (also for new analyzers); without them every relabelled unit is new
            labels_path = save_path / f"{file_prefix}labels.csv"
            if labels_path.is_file():
                original_labels = pd.read_csv(labels_path)
            else:
                original_labels = pd.DataFrame(columns=['quality', 'unit_id'])

            # with the metric names of the SpikeInterface version the model was trained with
            all_metrics = metrics_for_model(all_metrics, model_info)
            missing_metrics = [metric for metric in model_metric_names if metric not in all_metrics.columns]
            if len(missing_metrics) > 0:
                print(f"Skipping relabelled units of {target['path']}: it is missing metrics used by {model_folder.name}: {missing_metrics}")
                analyzers_missing_metrics[target['path']] = missing_metrics
                continue
            all_metrics = all_metrics[np.concat([model_metric_names, ['unit_id']])]
            # unit ids are matched as strings, since csv files are read back with pandas' inferred dtype
            all_metrics_unit_ids = all_metrics['unit_id'].astype(str).values
            original_qualities = dict(zip(original_labels['unit_id'].astype(str), original_labels['quality']))

            def model_labels_from(key):
                """This model's labels in the predictions saved for a model selection, as {unit id: label}."""
                predictions_path = save_path / predictions_file_name(file_prefix, key)
                if not predictions_path.is_file():
                    return {}
                predictions = pd.read_csv(predictions_path, index_col="unit_id")
                label_columns = [column for column in predictions.columns if column.endswith(f"{model_folder.name}_model_label")]
                if len(label_columns) == 0:
                    return {}
                return dict(zip(predictions.index.astype(str), predictions[label_columns[0]].fillna("").astype(str)))

            # what this model said about each unit: from the predictions of the selection the unit was picked
            # with, or else from this model's own saved predictions
            own_model_labels = model_labels_from(model_folder.name)
            picked_with_labels = {key: model_labels_from(key) for key in relabelled_units['picked_with'].unique()}
            already_learned = parent_trained_on.get(normalise_path(target['path']), set())
            analyzer_key = normalise_path(target['path'])

            new_units = []
            for _, relabelled_unit in relabelled_units.iterrows():
                new_class = _map_quality(relabelled_unit['quality'])
                if new_class is None:
                    num_not_in_classes += 1
                    continue
                unit_id = str(relabelled_unit['unit_id'])
                label_code = invert_label_conversion[new_class]
                row = row_of_unit.get((analyzer_key, unit_id)) if row_of_unit is not None else None
                if row is not None:
                    # the model was trained on this unit: compare with the label it learned, in the model's
                    # classes (e.g. MUA -> good is no change for a noise vs neural model)
                    if y_training[row] == label_code:
                        if training_rows['label_from'][row] == "":
                            num_same_as_original += 1
                        else:
                            # its label was set in an earlier retraining round
                            num_already_learned += 1
                        continue
                    changed_rows[row] = label_code
                    change = "label corrected"
                elif row_of_unit is not None:
                    # a unit the model was not trained on (e.g. from a new analyzer): a new row
                    new_units.append((label_code, unit_id))
                    change = "new unit"
                else:
                    # without training_units.csv, only the units and their original labels are known
                    if unit_id in already_learned and unit_id not in original_qualities:
                        # added in an earlier retraining round
                        num_already_learned += 1
                        continue
                    if unit_id in original_qualities and new_class == _map_quality(original_qualities[unit_id]):
                        num_same_as_original += 1
                        continue
                    new_units.append((label_code, unit_id))
                    change = "new unit"
                added_unit_details.append(dict(
                    analyzer=str(target['path']),
                    group="training" if target['analyzer_index'] is not None else "new",
                    unit_id=unit_id,
                    original_label=original_qualities.get(unit_id, ""),
                    model_label=picked_with_labels[relabelled_unit['picked_with']].get(unit_id, own_model_labels.get(unit_id, "")),
                    your_label=relabelled_unit['quality'],
                    taught_as=new_class,
                    change=change,
                ))
                units_added.setdefault(target['path'], []).append(unit_id)

            for label_code, unit_id in new_units:
                unit_metrics = all_metrics[all_metrics_unit_ids == unit_id]
                X_new.append(unit_metrics.values[0])
                X_just_values.append(unit_metrics.drop('unit_id', axis=1).values[0])
                y_new.append(label_code)
                new_rows_units.append((analyzer_key, unit_id))

        if len(y_new) == 0 and len(changed_rows) == 0:
            # say why, so the user knows what to do
            reasons = []
            if num_relabelled == 0:
                reasons.append("you have not relabelled any units yet")
            for path, missing_metrics in analyzers_missing_metrics.items():
                reasons.append(f"{Path(str(path)).parent.name}/{Path(str(path)).name} is missing metrics the model uses: "
                               f"{', '.join(missing_metrics)}")
            if num_not_in_classes > 0:
                reasons.append(f"{num_not_in_classes} units have a label this model doesn't predict "
                               f"(its classes are {', '.join(model_classes)})")
            if num_same_as_original > 0:
                reasons.append(f"{num_same_as_original} units got the same label as before, in this model's classes")
            if num_already_learned > 0:
                reasons.append(f"{num_already_learned} units were already added in an earlier retraining round")
            print(f"No relabelled units change the training data of {model_folder.name}, so it was not retrained: {reasons}")
            return dict(retrained=False, reasons=reasons, no_relabels=num_relabelled == 0)

        # the training data with your labels: corrected labels in their rows, then the new units
        y_corrected = y_training.copy()
        for row, label_code in changed_rows.items():
            y_corrected[row] = label_code
        # relabelled units in the held-out split are learned too, so they are not held out; both classifiers are
        # scored on the same held-out units, whose labels did not change
        test_idx = np.array([row for row in test_idx if row not in changed_rows], dtype=int)
        moved_idx = np.array(sorted(row for row in changed_rows if row not in set(train_idx)), dtype=int)
        X_test_proc, y_test = X_training_proc[test_idx], y_training[test_idx]

        def held_out_score(classifier):
            # no held-out units are left if you relabelled all of them
            return balanced_accuracy_score(y_test, classifier.predict(X_test_proc)) if len(y_test) > 0 else float("nan")

        # the classifier trained on the training split only, with the labels before this round: its score on the
        # held-out units is the baseline (the parent model itself may have been trained on the held-out units,
        # so it would not be a fair baseline)
        classifier_before = clone(model['classifier']).fit(X_training_proc[train_idx], y_training[train_idx])
        held_out_before = held_out_score(classifier_before)

        # then trained again with your labels
        X_new_proc = model_scaler.transform(
            model_imputer.transform(as_model_input(pd.DataFrame(X_just_values, columns=model_metric_names)))
        ) if len(y_new) > 0 else np.empty((0, X_training_proc.shape[1]))
        learned_idx = np.concatenate([train_idx, moved_idx])
        retrained_classifier = clone(model['classifier']).fit(
            np.concatenate([X_training_proc[learned_idx], X_new_proc]),
            np.concatenate([y_corrected[learned_idx], np.asarray(y_new, dtype=y_corrected.dtype)]),
        )

        all_training_data = pd.concat([
            pd.DataFrame(X_training_raw, columns=np.concat([['unit_id'],model_metric_names])),
            pd.DataFrame(X_new, columns=np.concat([model_metric_names,['unit_id']]))
        ])

        # save labels
        all_labels = pd.concat([
            pd.DataFrame({label_column: y_corrected}),
            pd.DataFrame({label_column: np.asarray(y_new, dtype=y_corrected.dtype)})
        ], ignore_index=True)
        all_labels.insert(0, 'unit_index', all_labels.index)

        retrained_model_folder.mkdir(exist_ok=True, parents=True)
        from sklearn.pipeline import Pipeline
        retrained_pipeline = Pipeline(
            [("imputer", model_imputer), ("scaler", model_scaler), ("classifier", retrained_classifier)]
        )
        dump(retrained_pipeline, retrained_model_folder / "best_model.skops")
        all_labels.to_csv(retrained_model_folder / 'labels.csv', index=False)
        all_training_data.to_csv(retrained_model_folder / 'training_data.csv', index=False)
        if training_rows is not None:
            # which round set the label of each row: none for the rows from training
            label_from = training_rows['label_from'].tolist()
            for row in changed_rows:
                label_from[row] = retrained_model_name
            pd.DataFrame({
                'analyzer': training_rows['analyzer'].tolist() + [analyzer for analyzer, _ in new_rows_units],
                'unit_id': training_rows['unit_id'].tolist() + [unit_id for _, unit_id in new_rows_units],
                'label_from': label_from + [retrained_model_name] * len(new_rows_units),
            }).to_csv(retrained_model_folder / TRAINING_UNITS_FILE, index=False)

        shutil.copyfile(model_folder / 'model_info.json', retrained_model_folder / 'model_info.json')

        # the retrained model learned from what its parent learned from, plus the relabelled units
        parent_units, parent_status = read_trained_on(model_folder)
        trained_on = {path: set(unit_ids) for path, unit_ids in parent_units.items()}
        for path, unit_ids in units_added.items():
            trained_on.setdefault(normalise_path(path), set()).update(unit_ids)
        write_trained_on(retrained_model_folder, trained_on, complete=(parent_status != "unknown"))

        self.project.model_paths = self.project.model_paths + [(retrained_model_folder, "local")]

        print(f"Successfully retrained model! Saved at {retrained_model_folder}.")
        held_out_balanced_accuracy = held_out_score(retrained_classifier)
        print(f"balanced accuracy (held-out test) = {held_out_before:.4f} without the relabelled units, "
              f"{held_out_balanced_accuracy:.4f} with them")

        # what this round did, for the Retraining history in section 4
        write_retraining_log(retrained_model_folder, dict(
            parent=model_folder.name,
            model=retrained_model_name,
            selection=selection_key,
            num_training_units_before=int(len(y_training)),
            num_new_units=len(y_new),
            num_corrected_labels=len(changed_rows),
            num_same_as_original=num_same_as_original,
            num_not_in_classes=num_not_in_classes,
            num_already_learned=num_already_learned,
            held_out=dict(num_units=int(len(y_test)), before=float(held_out_before), after=float(held_out_balanced_accuracy)),
        ), added_unit_details)

        return dict(name=retrained_model_name, num_units_added=len(added_unit_details), num_new_units=len(y_new),
                    num_corrected_labels=len(changed_rows), held_out_balanced_accuracy=held_out_balanced_accuracy,
                    held_out_before=held_out_before, num_held_out=int(len(y_test)), recordings=units_added)

    def show_curation_window(self, selected_directory, analyzer_index):

        analyzer_path = selected_directory

        print(f"\nLaunching SpikeInterface-GUI to curate analyzer at {analyzer_path}...")
        print("Label units as noise, good and MUA by pressing 'n', 'g' and 'm' on your keyboard.")
        print("Your labels are shown in the 'quality' column.")
        print("To change a label, select the unit and press another key: the new label replaces the old one.")
        print("Your labels are saved automatically when you close the SpikeInterface-GUI.")

        curate_filepath = Path(__file__).absolute().parent / "launch_sigui.py"
        self.run_sigui(
            [sys.executable, curate_filepath, analyzer_path, f'{self.output_folder}', f'{analyzer_index}'],
            title=f"Opening {Path(str(analyzer_path)).name} to do curation",
            steps=CURATE_STEPS,
            timing_key=f"curate:{analyzer_path}",
            on_closed=self.make_curation_button_list,
        )

    def run_sigui(self, command, title, steps, timing_key, on_closed):
        """
        Opens the SpikeInterface-GUI in its own process, without blocking this window. While it opens,
        a dialog shows the steps it goes through and the time it takes (and how long it took last time
        for this recording); once it is open, a banner says so. `on_closed` is called when it is closed.
        """
        if self.sigui_process is not None:
            print("The SpikeInterface-GUI is already open. Close it before opening another one.")
            return

        process = QProcess(self)
        process.setProcessChannelMode(QProcess.MergedChannels)
        environment = QProcessEnvironment.systemEnvironment()
        # print the SpikeInterface-GUI's messages as they come, not when it closes
        environment.insert("PYTHONUNBUFFERED", "1")
        environment.insert("PYTHONIOENCODING", "utf-8")
        process.setProcessEnvironment(environment)
        process.readyReadStandardOutput.connect(self.on_sigui_output)
        process.finished.connect(lambda exit_code, exit_status: self.on_sigui_finished(exit_code, on_closed))
        self.sigui_process = process
        self.sigui_window_shown = False
        self.sigui_cancelled = False
        self.sigui_output_buffer = ""
        self.sigui_labels_backups = []
        self.sigui_last_message = "No message was printed."
        self.sigui_timing_key = timing_key

        previous_seconds = self.project.config.get("sigui_opening_seconds", {}).get(timing_key)
        self.sigui_opening_dialog = OpeningDialog(title, steps, previous_seconds, parent=self)
        self.sigui_opening_dialog.cancelled.connect(self.cancel_sigui)
        self.sigui_opening_dialog.show()

        self.content_widget.setEnabled(False)

        process.start(str(command[0]), [str(argument) for argument in command[1:]])

    def on_sigui_output(self):
        output = bytes(self.sigui_process.readAllStandardOutput()).decode("utf-8", errors="replace")
        # output can arrive in pieces, so only complete lines are handled
        self.sigui_output_buffer += output.replace("\r\n", "\n")
        *lines, self.sigui_output_buffer = self.sigui_output_buffer.split("\n")
        for line in lines:
            if line.startswith(SIGUI_STEP_MARKER):
                self.sigui_opening_dialog.set_step(line[len(SIGUI_STEP_MARKER):].strip())
            elif line.strip() == SIGUI_SHOWN_MARKER:
                self.on_sigui_shown()
            elif line.startswith(LABELS_BACKUP_MARKER):
                backup, intended = line[len(LABELS_BACKUP_MARKER):].strip().split("|", 1)
                self.sigui_labels_backups.append((backup, intended))
            else:
                sys.stdout.write(line + "\n")
                if line.strip() != "":
                    self.sigui_last_message = line.strip()
        sys.stdout.flush()

    def on_sigui_shown(self):
        self.sigui_window_shown = True

        # remember how long it took, to show next time
        opening_seconds = self.project.config.setdefault("sigui_opening_seconds", {})
        opening_seconds[self.sigui_timing_key] = round(self.sigui_opening_dialog.elapsed_seconds(), 1)
        self.project.save_config()

        self.sigui_opening_dialog.done_opening()
        self.sigui_banner_text.setText(
            "<b>The SpikeInterface-GUI is open.</b> Label units with the <b>n</b>, <b>g</b> and <b>m</b> keys "
            "(noise, good, MUA) in the 'quality' column. To change a label, press another key: the new label "
            "replaces the old one.<br><b>Your labels are saved automatically when you close it</b>, and UnitRefine "
            "is available again."
        )
        self.sigui_banner.setVisible(True)

    def on_sigui_finished(self, exit_code, on_closed):
        if not self.sigui_window_shown:
            # still loading: cancelled, or failed before the window opened
            self.sigui_opening_dialog.done_opening()
            if exit_code != 0 and not self.sigui_cancelled:
                print("The SpikeInterface-GUI did not open. See the messages above for the reason.")
                QtWidgets.QMessageBox.warning(
                    self, "SpikeInterface-GUI did not open",
                    f"The SpikeInterface-GUI could not be opened:\n\n{self.sigui_last_message}\n\nThe terminal shows all the messages.",
                )
        self.sigui_process = None
        self.sigui_banner.setVisible(False)
        self.content_widget.setEnabled(True)
        if self.sigui_window_shown:
            print("SpikeInterface-GUI closed, resuming main app.\n")
        for backup, intended in self.sigui_labels_backups:
            QtWidgets.QMessageBox.warning(
                self, "Your labels were saved to a backup file",
                f"Your labels could not be saved to {intended} (it is read-only or open in another program), so "
                f"they were saved to:\n\n{backup}\n\nNothing is lost. Make {Path(intended).name} writable "
                "(in File Explorer: right-click, Properties, untick 'Read-only'), then copy the backup file over it.",
            )
        on_closed()

    def cancel_sigui(self):
        if self.sigui_process is not None:
            print("Cancelled opening the SpikeInterface-GUI.")
            self.sigui_cancelled = True
            self.sigui_process.kill()

    def show_train_window(self):

        if not self.curated_units_exist():
            print("No units have been curated - cannot train a model.")
            return

        self.w = TrainWindow(self.project)
        self.w.resize(800, 600)
        self.w.show()
        self.w.update_signal.connect(self.make_model_list)
        
    def curated_units_exist(self):
        
        for analyzer in self.project.analyzers.values():
            analyzer_folder = self.project.folder_name / analyzer['analyzer_in_project']
            labels_path = analyzer_folder / "labels.csv"
            if labels_path.is_file():
                return True

        return False


    def model_lineage(self, model_name):
        """A model and its retrained versions in the project, in order: the original, _retrained_01, _02, ..."""
        base_name, _ = _base_and_idx(model_name)
        versions = []
        for model_folder, _ in self.project.model_paths:
            name = Path(model_folder).name
            name_base, index = _base_and_idx(name)
            if name_base != base_name:
                continue
            # "base_retrained" without a number counts as the first retraining round
            round_number = 0 if name == base_name else (index if index is not None else 1)
            versions.append((round_number, Path(model_folder)))
        return base_name, [model_folder for _, model_folder in sorted(versions, key=lambda version: version[0])]

    def show_manage_models_window(self):

        if len(self.project.model_paths) == 0:
            print("No models trained or loaded into project.")
            return

        manage_window = ManageModelsWindow(self.project, parent=self)
        manage_window.models_changed.connect(self.make_model_list)
        manage_window.exec()

    def make_model_list(self):
        self.model_selector.set_models(self.project.model_paths)

    def models_to_retrain(self):
        """The names of the models chosen in 'Model to retrain' (one of the models chosen in 3.2, or all)."""
        names = self.retrain_choice.currentData() if self.retrain_choice.count() > 0 else None
        return list(names) if names is not None else self.model_selector.selected_model_names()

    def update_retrain_choice(self):
        """Fills 'Model to retrain' with the models chosen in 3.2; one model is retrained at a time."""
        names = self.model_selector.selected_model_names()
        previous = self.retrain_choice.currentData()
        self.retrain_choice.blockSignals(True)
        self.retrain_choice.clear()
        if self.model_selector.mode() == "cascade" and len(names) == 2:
            self.retrain_choice.addItem(f"Layer 1: {names[0]}", [names[0]])
            self.retrain_choice.addItem(f"Layer 2: {names[1]}", [names[1]])
        else:
            for name in names:
                self.retrain_choice.addItem(name, [name])
        for index in range(self.retrain_choice.count()):
            if self.retrain_choice.itemData(index) == previous:
                self.retrain_choice.setCurrentIndex(index)
        self.retrain_choice.setEnabled(self.retrain_choice.count() > 1)
        self.retrain_choice.blockSignals(False)
        self.on_retrain_choice_changed()

    def on_retrain_choice_changed(self):
        retrained_names = self.retrained_names()
        # a single model can be given any name; several models are named automatically
        if len(retrained_names) == 1:
            self.retrainedModelNameForm.setText(retrained_names[0])
            self.retrainedModelNameForm.setReadOnly(False)
        else:
            self.retrainedModelNameForm.setText(", ".join(retrained_names))
            self.retrainedModelNameForm.setReadOnly(True)

    def retrained_names(self):
        """The next free retrained name (…_retrained_01, _02, …) for each model to retrain."""

        existing_names = [str(model_path.name) for (model_path, _) in self.project.model_paths]
        retrained_names = []
        for model_name in self.models_to_retrain():
            base_name, _ = _base_and_idx(model_name)
            retrained_name = _next_retrained_name(base_name, existing_names + retrained_names, width=2)
            retrained_names.append(retrained_name)
        return retrained_names

    def on_model_selection_changed(self):

        self.update_retrain_choice()

        self.make_validate_button_list()
        self.make_relabel_button_list()

    def show_validate_window(self, analyzer, analyzer_index, relabel=False, *, save_folder=None, file_prefix=""):

        analyzer_path = analyzer['path']
        if save_folder is None:
            save_folder = Path(self.output_folder) / analyzer['analyzer_in_project']

        problem = self.model_selector.describe_problem()
        if problem is not None:
            print(problem)
            return

        mode = self.model_selector.mode()
        model_folders = [str(model_folder) for model_folder, _ in self.model_selector.selected_models()]
        selection_key = self.model_selector.selection_key()

        validate_filepath = Path(__file__).absolute().parent / "launch_sigui_validate.py"
        command = [
            sys.executable, validate_filepath, str(analyzer_path),
            "--save_folder", str(save_folder), "--file_prefix", file_prefix,
            "--mode", mode, "--model_folders", *model_folders, "--selection_key", selection_key,
        ]
        # curated or imported labels fill the quality column: of a project analyzer, or imported for a new one
        human_labels_path = Path(save_folder) / f"{file_prefix}labels.csv"
        if human_labels_path.is_file():
            command += ["--human_labels", str(human_labels_path)]
        if relabel:
            units_file = units_to_relabel_path(dict(save_folder=save_folder, file_prefix=file_prefix), selection_key)
            if not units_file.is_file():
                print("Click 'Find least confident units' first, to choose the units to relabel.")
                return
            command += ["--relabel", "--units_file", str(units_file)]
        purpose = "to relabel the least confident units" if relabel else "to inspect the model predictions"
        self.run_sigui(
            command,
            title=f"Opening {Path(str(analyzer_path)).name} {purpose}",
            steps=INSPECT_STEPS,
            timing_key=f"inspect:{analyzer_path}",
            on_closed=self.make_relabel_button_list,
        )

def main():
    """Starts UnitRefine; see launcher.py, which shows an 'Opening UnitRefine' box while it loads."""
    from unitrefine.launcher import main as launch
    launch()

def initial_window_size(available_geometry):
    """
    The window opens wide enough for the sections at their maximum width (or the whole screen width on a
    small laptop), and as tall as the screen allows; the rest scrolls.
    """
    width = min(CONTENT_MAX_WIDTH + 60, available_geometry.width())
    height = min(1100, int(available_geometry.height() * 0.9))
    return QSize(width, height)

def is_a_model(directory):
    return (Path(directory) / "best_model.skops").is_file()

def _base_and_idx(name: str):
    """
    Returns (base_name, idx or None) where idx is the numeric suffix after _retrained_.
    Handles:
      - foo
      - foo_retrained
      - foo_retrained_01
      - foo_retrained_9
    """
    m = _RETRAINED_RE.match(name)
    if not m:
        return name, None
    base = m.group("base")
    idx = m.group("idx")
    return base, (int(idx) if idx is not None else None)

def _next_retrained_name(base: str, existing_names: list[str], width: int = 2) -> str:
    """
    Finds next available retrained index based on existing names.
    Produces: base_retrained_01, base_retrained_02, ...
    If base_retrained exists without number, treat it as idx=1.
    """
    max_idx = 0
    for n in existing_names:
        b, idx = _base_and_idx(n)
        if b != base:
            continue

        if n == f"{base}_retrained" and idx is None:
            max_idx = max(max_idx, 1)
        elif idx is not None:
            max_idx = max(max_idx, idx)

    next_idx = max_idx + 1
    return f"{base}_retrained_{next_idx:0{width}d}"

if __name__ == "__main__":
    main()
