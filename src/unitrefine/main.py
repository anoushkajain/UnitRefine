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
from modAL.models import ActiveLearner
from modAL.uncertainty import uncertainty_sampling
from spikeinterface.curation.model_based_curation import load_model
from skops.io import dump

import PyQt5.QtWidgets as QtWidgets
from PyQt5.QtCore import Qt
from PyQt5.QtGui import QFont, QIcon
from PyQt5.QtWidgets import QStyleFactory

from spikeinterface_gui.main import check_folder_is_analyzer
from spikeinterface.core.core_tools import is_path_remote
from spikeinterface.core import load_sorting_analyzer

from unitrefine.train import TrainWindow

import warnings
warnings.filterwarnings("ignore")

class UrlInputDialog(QtWidgets.QDialog):
    def __init__(self, parent=None, default=None, background_color=None):
        super().__init__(parent)

        if default is None:
            default = "https://www.example.com"
        
        self.setWindowTitle("Open URL")
        self.setMinimumWidth(400)

        self.setStyleSheet(f"background-color: '#{background_color}'")

        layout = QtWidgets.QVBoxLayout()

        self.label = QtWidgets.QLabel("Please enter the s3 path to an analyzer:")
        layout.addWidget(self.label)

        self.url_input = QtWidgets.QLineEdit()
        self.url_input.setPlaceholderText(default)
        self.url_input.setClearButtonEnabled(True)
        layout.addWidget(self.url_input)
        self.url_input.setStyleSheet("background-color: '#FFFFFF'")

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
            json.dump(self.config, f)

        for analyzer_name, analyzer_dict in self.analyzers.items():

            path = analyzer_dict.get('path')

            save_path = Path(folder_name / "analyzers" / f"analyzer_{analyzer_name}")
            save_path.mkdir(exist_ok=True)

            if path is not None:
                with open(save_path / 'path.txt', 'w') as output:
                    output.write(path)

            curation = analyzer_dict.get('labels')
            if curation is not None:
                curation.to_csv(save_path / 'labels.csv')

    def add_analyzer(self, directory):

        analyzer_keys = self.analyzers.keys()
        if len(analyzer_keys) > 0:
            max_key = max(list(analyzer_keys))
            new_key = max_key + 1
        else:
            new_key = 0

        analyer_in_project = Path(f'analyzers/analyzer_{new_key}')
        self.analyzers[new_key] = {'path': directory, 'analyzer_in_project': analyer_in_project}


def load_project(folder_name):

    folder_name = Path(folder_name)

    project = UnitRefineProject(folder_name)

    analyzers_folder = folder_name / "analyzers"
    analyzer_folders = list(analyzers_folder.glob('*/'))

    for analyzer_folder in analyzer_folders:

        analyzer_dict = {}

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
    model_folders = [folder for folder in list(models_folder.glob('*/')) if '.DS' not in str(folder)]

    for model_folder in model_folders:
        if (model_folder / "hfh_path.txt").is_file():
            project.model_paths.append((model_folder, "hfh"))
        else:
            project.model_paths.append((model_folder, "local"))

    return project

class MainWindow(QtWidgets.QWidget):

    def __init__(self, project):
        
        super().__init__()

        self.setStyleSheet("background-color: white")
        
        self.w = None
        self.sorting_analyzer_paths = []
        self.curate_buttons = []
        self.delete_buttons = []

        self.output_folder = project.folder_name

        self.project = project
        self.percentage_float = 20

        self.main_layout = QtWidgets.QGridLayout(self)
        self.retrainedModelNameForm = None

        to_curateWidget = QtWidgets.QWidget()
        to_curateWidget.setStyleSheet("background-color: LightBlue")

        projectWidget = QtWidgets.QWidget()
        projectWidget.setStyleSheet("background-color: LightBlue")

        projectLayout = QtWidgets.QGridLayout()

        projectTitleWidget = QtWidgets.QLabel("1. PROJECT DETAILS")
        projectTitleWidget.setStyleSheet("font-weight: bold; font-size: 20pt;")
        projectLayout.addWidget(projectTitleWidget, 0, 0)

        output_folder_text = QtWidgets.QLabel(f"Project folder: {self.output_folder}")
        projectLayout.addWidget(output_folder_text,1,0,1,3)


        labels_text = QtWidgets.QLabel("Labels: ")
        labels_text.setAlignment(Qt.AlignmentFlag.AlignRight) 
        projectLayout.addWidget(labels_text,2,0,1,1)
        self.change_labels_button = QtWidgets.QLineEdit("noise, good, MUA")
        projectLayout.addWidget(self.change_labels_button,2,1,1,2)

        projectWidget.setLayout(projectLayout)
        self.main_layout.addWidget(projectWidget)

        ###############
        # CURATE
        ##############

        saWidget = QtWidgets.QWidget()
        saWidget.setStyleSheet("background-color: '#CBEECB'")

        saLayout = QtWidgets.QGridLayout()
        
        curation_title_text = "2. CURATION"

        curationTitleWidget = QtWidgets.QLabel(curation_title_text)
        curationTitleWidget.setStyleSheet("font-weight: bold; font-size: 20pt;")
        saLayout.addWidget(curationTitleWidget,0,0,1,1)
        
        self.add_sa_button = QtWidgets.QPushButton("+ Add Sorting Analyzer Folder")
        self.add_sa_button.clicked.connect(self.selectDirectoryDialog)
        saLayout.addWidget(self.add_sa_button,1,0)

        self.add_s3_button = QtWidgets.QPushButton("+ Add Analyzer from s3")
        self.add_s3_button.clicked.connect(self.add_from_s3)
        saLayout.addWidget(self.add_s3_button,1,1,1,2)

        self.curate_text = QtWidgets.QLabel("Curated?")
        saLayout.addWidget(self.curate_text,2,2)

        self.saLayout = saLayout
        saWidget.setLayout(saLayout)
        self.make_curation_button_list()

        ###############
        # TRAIN
        ##############

        trainWidget = QtWidgets.QWidget()
        trainWidget.setStyleSheet("background-color: PeachPuff")
        self.validateLayout = QtWidgets.QGridLayout()
        self.trainLayout = QtWidgets.QGridLayout()

        trainTitleWidget = QtWidgets.QLabel("3. TRAIN or LOAD models")
        trainTitleWidget.setStyleSheet("font-weight: bold; font-size: 20pt;")
        self.trainLayout.addWidget(trainTitleWidget, 0, 0)

        train_button = QtWidgets.QPushButton("Train")
        train_button.clicked.connect(self.show_train_window)
        self.trainLayout.addWidget(train_button, 1, 0, 1, 2)

        load_model_button = QtWidgets.QPushButton("+ Load")
        load_model_button.clicked.connect(self.selectModelDialog)
        self.trainLayout.addWidget(load_model_button, 1, 2, 1, 1)

        load_model_hf = QtWidgets.QPushButton("+ Load from HFH")
        load_model_hf.clicked.connect(self.add_from_hfh)
        self.trainLayout.addWidget(load_model_hf, 1, 3, 1, 1)

        trainTitleWidget2 = QtWidgets.QLabel("After training or loading, choose your model:")
        self.trainLayout.addWidget(trainTitleWidget2, 2, 0, 1, 3)

        validateTitleWidget = QtWidgets.QLabel("And inspect its predictions on an analyzer:")
        self.validateLayout.addWidget(validateTitleWidget, 0, 0, 1, 3 )


        self.trainANDvalidateLayout = QtWidgets.QVBoxLayout(trainWidget)
        self.trainANDvalidateLayout.addLayout(self.trainLayout)
        self.trainANDvalidateLayout.addLayout(self.validateLayout)

        self.make_model_list()

        self.modelLayout = QtWidgets.QGridLayout()

        self.main_layout.addWidget(saWidget)
        self.main_layout.addWidget(trainWidget)

        #################
        # RETRAIN WIDGET
        ##################

        retrainWidget = QtWidgets.QWidget()
        retrainWidget.setStyleSheet("background-color: Pink")

        self.retrainLayout = QtWidgets.QGridLayout()
        self.relabelLayout = QtWidgets.QGridLayout()

        retrainTitleWidget = QtWidgets.QLabel("4. RELABEL and RETRAIN")
        retrainTitleWidget.setStyleSheet("font-weight: bold; font-size: 20pt;")
        self.relabelLayout.addWidget(retrainTitleWidget,0,0,1,2)
        self.relabelLayout.addWidget(QtWidgets.QLabel("Relabel uncertain units from each analyzer..."),2,0,1,1)

        
        self.percentageOfUnitsForm = QtWidgets.QLineEdit(f"{self.percentage_float}")
        self.percentageOfUnitsForm.setStyleSheet("background-color: white;")
        self.percentageOfUnitsForm.textChanged.connect(self.update_percentage_float)
        self.unitLabel = QtWidgets.QLabel("Percentage of least confident units to display:")
        
        self.relabelLayout.addWidget(self.percentageOfUnitsForm,1,1,1,1)
        self.relabelLayout.addWidget(self.unitLabel,1,0,1,1)

        retrain_text = QtWidgets.QLabel("Retrained model name: ")
        self.retrainLayout.addWidget(retrain_text,1,0,1,1)

        current_model_name = self.combo_box.currentText()
        self.retrainedModelNameForm = QtWidgets.QLineEdit(f"{current_model_name}_retrained")
        self.retrainedModelNameForm.setStyleSheet("background-color: white; color: black;")
        self.retrainLayout.addWidget(self.retrainedModelNameForm,1,1,1,1)

        retrain_button = QtWidgets.QPushButton('Retrain model')
        retrain_button.clicked.connect(partial(self.retrain_model))
        self.retrainLayout.addWidget(retrain_button,2,0,1,2)

        self.relabelANDvalidateLayout = QtWidgets.QVBoxLayout(retrainWidget)
        self.relabelANDvalidateLayout.addLayout(self.relabelLayout)
        self.relabelANDvalidateLayout.addLayout(self.retrainLayout)

        self.make_relabel_button_list()

        self.main_layout.addWidget(saWidget)
        self.main_layout.addWidget(trainWidget)
        self.main_layout.addWidget(retrainWidget)


        ###############
        # CODE BUTTON
        ###############

        apply_code_button = QtWidgets.QPushButton("Generate code to apply model to analyzer")
        apply_code_button.clicked.connect(self.make_apply_code)
        self.main_layout.addWidget(apply_code_button)

    def add_from_s3(self):
        
        dialog = UrlInputDialog(background_color="CBEECB")

        if dialog.exec():
            url = dialog.get_url()
            if is_path_remote(url):
                self.project.add_analyzer(url)
                self.project.save()
                self.make_curation_button_list()
                self.make_validate_button_list()
            else:
                print(f"url {url} is not a valid analyzer path.")

    def update_percentage_float(self):

        percentage_text = self.percentageOfUnitsForm.text()
        try:
            percentage_float = float(percentage_text)
            
            if percentage_float < 1:
                print("Percentage of units displayed should be larger than 1. Please change the value.")
                return
            if percentage_float > 100:
                print("Percentage of units displayed should be smaller than 100.  Please change the value.")
                return 
            self.percentage_float = percentage_float
        except:
            print("Cannot parse percentage value. Must be a number. Please change the value.")
            return
        
        self.make_relabel_button_list()

    def add_from_hfh(self):
        
        dialog = UrlInputDialog(default = 'AnoushkaJain3/UnitRefine-mice-sua-classifier', background_color="FFDAB9")

        if dialog.exec():
            url = dialog.get_url()
            if repo_exists(url):
                model_directory = self.project.folder_name / "models" / url.split('/')[-1]
                model_directory.mkdir(exist_ok=True)

                self.project.model_paths = self.project.model_paths + [(model_directory, "hfh")]

                from huggingface_hub import snapshot_download

                print(f"\nDownloading model from HuggingFaceHub repo {url}.\n")

                QtWidgets.QApplication.setOverrideCursor(Qt.CursorShape.WaitCursor)

                try:
                    snapshot_download(url, local_dir=model_directory)
                finally:
                    QtWidgets.QApplication.restoreOverrideCursor()

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

                model_info_path = model_directory / "model_info.json"
                with open(model_info_path, 'w') as f:
                    json.dump(model_info, f, indent=4)

                self.make_model_list()

            else:
                print(f"Repo {url} does not exist.")

    def make_apply_code(self):

        code_text = "\n"
        code_text += "import spikinterface.full as si\n\n"
        code_text += "# point this path to the analyzer you want to apply the model to\n"
        code_text += "path_to_analyzer = 'path/to/analyzer'\n"
        code_text += "analyzer_to_curate = si.load_sorting_analyzer(path_to_analyzer)\n\n"

        code_text += f"model_folder = {self.project.selected_model}\n\n"
        code_text += "# labels will be a list of curated labels, determined by the model.\n"
        code_text += "labels = si.auto_label_units(\n\tsorting_analyzer = analyzer_to_curate,\n\tmodel_folder = model_folder,\n)\n\n"
        code_text += "Read more here: https://spikeinterface.readthedocs.io/en/stable/tutorials/curation/plot_1_automated_curation.html\n\n"      
            
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
                self.project.model_paths = self.project.model_paths + [(selected_directory, "local")]
                self.make_model_list()
            else:
                print(f"{selected_directory} is not a UnitRefine model folder.")
            
    def make_curation_button_list(self):

        for widget_no in range(3, self.saLayout.count()):
            self.saLayout.itemAt(widget_no).widget().deleteLater()

        for analyzer_index, analyzer in self.project.analyzers.items():

            selected_directory = analyzer['path']

            if len(str(selected_directory)) > 50:
               selected_directory_text_display = "..." + str(selected_directory)[-50:]
            else:
                selected_directory_text_display = selected_directory

            curate_button = QtWidgets.QPushButton(f'Curate "{selected_directory_text_display}"')
            curate_button.clicked.connect(partial(self.show_curation_window, selected_directory, analyzer_index))
            self.saLayout.addWidget(curate_button,4+analyzer_index,0)

            self.btn_settings = QtWidgets.QToolButton()

            icon = self.style().standardIcon(QtWidgets.QStyle.SP_TrashIcon) 
            self.btn_settings.setIcon(icon)
            self.btn_settings.setToolTip("Remove from curation")
            self.btn_settings.clicked.connect(partial(self.remove_analyzer, analyzer_index))

            self.saLayout.addWidget( self.btn_settings,4+analyzer_index,2)
    
            curation_output_folder = Path(self.project.folder_name) / Path(f"analyzers/analyzer_{analyzer_index}")
            curation_output_folder.mkdir(exist_ok=True)

            if (curation_output_folder / "all_metrics.csv").is_file() and (curation_output_folder / "labels.csv").is_file():
                just_labels = pd.read_csv(curation_output_folder / "labels.csv")
                all_metrics = pd.read_csv(curation_output_folder / "all_metrics.csv")
                not_curated_text = QtWidgets.QLabel(f"{len(just_labels)}/{len(all_metrics)}")

            else:
                not_curated_text = QtWidgets.QLabel("---")

            self.saLayout.addWidget(not_curated_text,4+analyzer_index,1)

    def remove_analyzer(self, analyzer_index):

        analyzer_indices = list(self.project.analyzers.keys())

        analyzer_folder = Path(self.project.folder_name) / Path(self.project.analyzers[analyzer_indices[analyzer_index]]['analyzer_in_project'])
        shutil.rmtree(str(analyzer_folder))
        if analyzer_folder.is_dir():
            os.rmdir(str(analyzer_folder))

        self.project.analyzers.pop(analyzer_indices[analyzer_index])

        self.make_curation_button_list()
        self.make_validate_button_list()
        self.make_relabel_button_list()

    def make_relabel_button_list(self):

        for widget_no in range(4, self.relabelLayout.count()):
            self.relabelLayout.itemAt(widget_no).widget().deleteLater()

        for analyzer_index, analyzer in enumerate(self.project.analyzers.values()):

            selected_directory = analyzer['path']

            if len(str(selected_directory)) > 40:
               selected_directory_text_display = "..." + str(selected_directory)[-40:]
            else:
                selected_directory_text_display = selected_directory

            curate_button = QtWidgets.QPushButton(f'Relabel "{selected_directory_text_display}"')
            curate_button.clicked.connect(partial(self.show_validate_window, analyzer, analyzer_index, True))

            button_text = "---"
            if self.project.selected_model is not None:
                relabelled_units_path = self.project.folder_name / analyzer['analyzer_in_project'] / f"relabelled_units_{self.project.selected_model.name}.csv"    
                all_metrics_path = self.project.folder_name / analyzer['analyzer_in_project'] / "all_metrics.csv"                     
                if all_metrics_path.is_file() and relabelled_units_path.is_file():
                    all_metrics = pd.read_csv(all_metrics_path)
                    labels = pd.read_csv(relabelled_units_path)
                    button_text = f"{len(labels)}/{int(np.ceil(len(all_metrics)*self.percentage_float/100))}"

            self.relabelLayout.addWidget(curate_button,3+analyzer_index,0,1,3)
            self.relabelLayout.addWidget(QtWidgets.QLabel(button_text),3+analyzer_index,4,1,3)

    def make_validate_button_list(self):

        for widget_no in range(1, self.validateLayout.count()):
           self.validateLayout.itemAt(widget_no).widget().deleteLater()

        for analyzer_index, analyzer in enumerate(self.project.analyzers.values()):

            selected_directory = analyzer['path']

            if len(str(selected_directory)) > 40:
               selected_directory_text_display = "..." + str(selected_directory)[-40:]
            else:
                selected_directory_text_display = selected_directory

            curate_button = QtWidgets.QPushButton(f'Inspect "{selected_directory_text_display}"')
            curate_button.clicked.connect(partial(self.show_validate_window, analyzer, analyzer_index, False))

            
            self.validateLayout.addWidget(curate_button,1+analyzer_index,0,1,3)


    def retrain_model(self):

        retrained_model_name = self.retrainedModelNameForm.text()

        current_model_name = self.combo_box.currentText()
        self.project.selected_model, hfh_or_local = [model for model in self.project.model_paths if str(current_model_name) == str(model[0].name)][0]

        retrained_model_folder = self.project.selected_model.parent / retrained_model_name

        model, model_info = load_model(
            model_folder=self.project.selected_model, trust_model=True
        )
        label_conversion = model_info['label_conversion']
        invert_label_conversion = {val: int(key) for key, val in label_conversion.items()}
        model_metric_names = model.feature_names_in_

        model_imputer = model['imputer']
        model_imputer.keep_empty_features = True

        model_scaler = model['scaler']

        X_training_raw = pd.read_csv(Path(self.project.selected_model) / 'training_data.csv')
        X_training = X_training_raw.drop('unit_id', axis=1).values
        Y_training_raw = pd.read_csv(Path(self.project.selected_model) / 'labels.csv')
        Y_training = Y_training_raw.drop('unit_index', axis=1).values

        learner = ActiveLearner(
            estimator=model['classifier'],                    
            query_strategy=uncertainty_sampling,   # Query strategy: select most uncertain samples
            X_training=X_training,              
            y_training=Y_training.ravel()          
        )

        X_new = []
        X_just_values = []
        y_new = []

        for analyzer_name, analyzer_dict in self.project.analyzers.items():

            folder_name = self.project.folder_name
            save_path = Path(folder_name / "analyzers" / f"analyzer_{analyzer_name}")

            relabelled_units_path = save_path / f"relabelled_units_{self.project.selected_model.name}.csv"
            if relabelled_units_path.is_file():
                relabelled_units = pd.read_csv(save_path / f"relabelled_units_{self.project.selected_model.name}.csv")
            else:
                relabelled_units = pd.DataFrame(columns=['quality', 'unit_id'])
            all_metrics_path = save_path / "all_metrics.csv"
            if all_metrics_path.is_file():
                all_metrics = pd.read_csv(all_metrics_path)
            else:
                analyzer = load_sorting_analyzer(analyzer_dict['path'], load_extensions=False)
                analyzer.load_extension("quality_metrics")
                analyzer.load_extension("template_metrics")
                if analyzer.has_extension('template_metrics'):
                    tms = analyzer.get_extension("template_metrics").get_data()
                else:
                    tms = pd.DataFrame()
                
                if analyzer.has_extension('quality_metrics'):
                    qms = analyzer.get_extension("quality_metrics").get_data()
                else:
                    qms = pd.DataFrame()
                
                all_metrics = pd.concat([qms, tms], axis=1)
                all_metrics['unit_id'] = all_metrics.index
            labels_path = save_path / "labels.csv"
            if labels_path.is_file():
                original_labels = pd.read_csv(save_path / "labels.csv")
            else:
                original_labels = pd.DataFrame(columns=['quality', 'unit_id'])

            all_metrics = all_metrics[np.concat([model_metric_names, ['unit_id']])]

            y_new_with_unit_ids = []
            for _, relabelled_unit in relabelled_units.iterrows():
                if relabelled_unit['unit_id'] in original_labels['unit_id'].values:
                    if relabelled_unit['quality'] != original_labels.query(f"unit_id == {relabelled_unit['unit_id']}")['quality'].values[0]:
                        # New label contradicts the original label
                        y_new_with_unit_ids.append([invert_label_conversion[relabelled_unit['quality']], relabelled_unit['unit_id']])
                else:
                    # A newly labelled unit
                    y_new_with_unit_ids.append([invert_label_conversion[relabelled_unit['quality']], relabelled_unit['unit_id']])


            for quality, unit_id in y_new_with_unit_ids:
                X_new.append(all_metrics.query(f'unit_id == {unit_id}').values[0])
                X_just_values.append(all_metrics.query(f'unit_id == {unit_id}').drop('unit_id', axis=1).values[0])
                y_new.append(quality)

        X_imputed = model_imputer.fit_transform(pd.DataFrame(X_just_values,columns=model_metric_names))
        learner.teach(X_imputed, y_new)

        all_training_data = pd.concat([
            pd.DataFrame(X_training_raw, columns=np.concat([['unit_id'],model_metric_names])),
            pd.DataFrame(X_new, columns=np.concat([model_metric_names,['unit_id']]))
        ])

        # save labels
        all_labels = pd.concat([
            Y_training_raw,
            pd.DataFrame(y_new, columns=['0'])
        ])
        all_labels['unit_index'] = all_labels.index

        retrained_model_folder.mkdir(exist_ok=True, parents=True)
        from sklearn.pipeline import Pipeline
        retrained_pipeline = Pipeline(
            [("imputer", model_imputer), ("scaler", model_scaler), ("classifier", learner.estimator)]
        )
        dump(retrained_pipeline, retrained_model_folder / "best_model.skops")
        all_labels.to_csv(retrained_model_folder / 'labels.csv', index=False)
        all_training_data.to_csv(retrained_model_folder / 'training_data.csv', index=False)

        shutil.copyfile(self.project.selected_model / 'model_info.json', retrained_model_folder / 'model_info.json')

        self.project.model_paths = self.project.model_paths + [(retrained_model_folder, "local")]
        
        self.make_model_list()
        
        print(f"Successfully retrained model! Saved at {retrained_model_folder}.")
        print(f"balanced accuracy = {learner.score(all_training_data.drop('unit_id', axis=1).values, all_labels['0'].values)}")

    def show_curation_window(self, selected_directory, analyzer_index):

        self.change_labels_button.setReadOnly(True)
        self.change_labels_button.setStyleSheet("background-color: LightBlue")

        analyzer_path = selected_directory

        print(f"\nLaunching SpikeInterface-GUI to curate analyzer at {analyzer_path}...")
        print("Label units as noise, good and MUA by pressing 'n', 'g' and 'm' on your keyboard.")

        curate_filepath = Path(__file__).absolute().parent / "launch_sigui.py"
        subprocess.run([sys.executable, curate_filepath, analyzer_path, f'{self.output_folder}', f'{analyzer_index}'])
        print("SpikeInterface-GUI closed, resuming main app.\n")

        self.make_curation_button_list()

    def show_train_window(self):
        self.w = TrainWindow(self.project)
        self.w.resize(800, 600)
        self.w.show()
        self.w.update_signal.connect(self.make_model_list)
        

    def make_model_list(self):
        self.combo_box = QtWidgets.QComboBox(self)
        self.combo_box.currentIndexChanged.connect(self.update_retrained_name)
        model_folders = [Path(model[0]) for model in self.project.model_paths]
        model_names = [model_folder.name for model_folder in model_folders]
        self.combo_box.addItems(model_names)       
        self.trainLayout.addWidget(self.combo_box,3,0,1,4)
        new_model_index = self.combo_box.count() - 1
        self.combo_box.setCurrentIndex(new_model_index)
        self.update_retrained_name()

    def update_retrained_name(self):
        if self.retrainedModelNameForm is not None:
            self.retrainedModelNameForm.setText(f"{self.combo_box.currentText()}_retrained")
        current_model_name = self.combo_box.currentText()
        self.project.selected_model, hfh_or_local = [model for model in self.project.model_paths if str(current_model_name) == str(model[0].name)][0]
        self.make_validate_button_list()
        try:
            self.make_relabel_button_list()
        except:
            return

    def show_validate_window(self, analyzer, analyzer_index, relabel=False):

        analyzer_path = analyzer['path']
        
        current_model_name = self.combo_box.currentText()
        self.project.selected_model, hfh_or_local = [model for model in self.project.model_paths if str(current_model_name) == str(model[0].name)][0]

        analyzer_in_project = analyzer['analyzer_in_project']

        validate_filepath = Path(__file__).absolute().parent / "launch_sigui_validate.py"
        subprocess.run([sys.executable, validate_filepath, str(analyzer_path), str(self.output_folder), str(analyzer_in_project), str(analyzer_index), self.project.selected_model, current_model_name, hfh_or_local, str(relabel), str(self.percentage_float)])
        print("SpikeInterface-GUI closed, resuming main app.")
        self.make_relabel_button_list()

def main():
        
    parser = ArgumentParser(
        description="UnitRefine - curate your sorting and create a machine learning model based on your curation."
    )
    parser.add_argument(
        "--project_folder",
        required=True,
        type=str,
    )

    args = parser.parse_args()        
    project_folder = Path(args.project_folder).resolve()

    if project_folder.is_dir():
        print("Project already exists. Loading...")
        project = load_project(project_folder)
    else:
        print("Project Folder does not exist. Creating now...")
        project = UnitRefineProject(project_folder)
        project.save()

    app = QtWidgets.QApplication(sys.argv)
    app.setStyle(QStyleFactory.create("Fusion"))
    app.setStyleSheet("* { color: black; }")
    icon_file = Path(__file__).absolute().parent / 'resources' / 'logo.png'
    if icon_file.exists():
        app.setWindowIcon(QIcon(str(icon_file)))

    custom_font = QFont()
    custom_font.setFamily("courier new")
    app.setFont(custom_font)
    w = MainWindow(project)
    w.setWindowTitle('UnitRefine')
    w.show()
    app.exec()

def is_a_model(directory):
    return (Path(directory) / "best_model.skops").is_file()

if __name__ == "__main__":
    main()
