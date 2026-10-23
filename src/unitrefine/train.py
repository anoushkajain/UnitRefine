from datetime import datetime
from functools import partial

from pathlib import Path

import pandas as pd
import PyQt5.QtWidgets as QtWidgets
from PyQt5.QtCore import pyqtSignal
from PyQt5.QtCore import Qt
from spikeinterface.curation import train_model
import numpy as np
from copy import copy

from unitrefine.evaluation import write_trained_on
from unitrefine.predictions import as_model_input
from unitrefine.theme import make_primary
from unitrefine.widgets import BusyNotice

TEST_PREDICTIONS_FILE = "test_predictions.csv"


def save_test_predictions(model_folder, test_size, seed):
    """
    Saves the model's labels for the units held out from training (test_predictions.csv), for the figures.
    train_model does not save which units it held out, so the split is made again in the same way (same
    data, test size and seed). It is only saved if it gives the balanced accuracy train_model reported,
    so the figures never show a different test set from the one the model was scored on.
    """
    from sklearn.metrics import balanced_accuracy_score
    from sklearn.model_selection import train_test_split
    from spikeinterface.curation.model_based_curation import load_model

    model_folder = Path(model_folder)
    pipeline, model_info = load_model(model_folder=model_folder, trust_model=True)
    # as 32-bit floats, like train_model: 64-bit values give a slightly different test score
    metrics = as_model_input(pd.read_csv(model_folder / "training_data.csv", index_col="unit_id"))
    label_codes = pd.read_csv(model_folder / "labels.csv", index_col="unit_index").iloc[:, 0]

    _, test_metrics, _, test_codes = train_test_split(
        metrics, label_codes, test_size=test_size, random_state=seed, stratify=label_codes
    )
    predicted_codes = pipeline.predict(test_metrics[list(pipeline.feature_names_in_)])

    reported = pd.read_csv(model_folder / "model_accuracies.csv")["balanced_accuracy"].values[0]
    reproduced = balanced_accuracy_score(test_codes, predicted_codes)
    # model_accuracies.csv has 4 decimals
    if abs(reproduced - reported) > 6e-5:
        print(f"Could not reproduce the test set of {model_folder.name}, so its test results are not shown in the figures.")
        return

    label_conversion = model_info["label_conversion"]
    pd.DataFrame({
        "human": [label_conversion[str(int(code))] for code in test_codes],
        "predicted": [label_conversion[str(int(code))] for code in predicted_codes],
    }).to_csv(model_folder / TEST_PREDICTIONS_FILE, index=False)

def spikeinterface_metric_names():
    """The quality and template metric names of the installed SpikeInterface version."""
    try:
        from spikeinterface.metrics.quality import ComputeQualityMetrics
        from spikeinterface.metrics.template import ComputeTemplateMetrics
    except ImportError:
        # SpikeInterface versions before 0.104
        from spikeinterface.qualitymetrics import ComputeQualityMetrics
        from spikeinterface.postprocessing import ComputeTemplateMetrics
    return list(ComputeQualityMetrics.get_metric_columns()), list(ComputeTemplateMetrics.get_metric_columns())


_quality_metric_names, _template_metric_names = spikeinterface_metric_names()

# the presets use the metric names of the installed SpikeInterface version, so they match the analyzers' metrics
metrics_presets = {
    "Spike times metrics": [name for name in ["num_spikes", "firing_rate", "presence_ratio", "snr", "isi_violations_ratio",
                                              "isi_violations_count", "rp_contamination", "rp_violations",
                                              "sliding_rp_violation", "sync_spike_2", "sync_spike_4", "sync_spike_8",
                                              "firing_range"] if name in _quality_metric_names],
    "Quality metrics": _quality_metric_names,
    "Template metrics": _template_metric_names,
}

# The classifiers, scalers and imputers train_model can try; every combination ticked is trained, and the best is kept.
# (name, description, package it needs if not scikit-learn)
CLASSIFIER_OPTIONS = [
    ("RandomForestClassifier", "Random forest", None),
    ("GradientBoostingClassifier", "Gradient boosting", None),
    ("AdaBoostClassifier", "AdaBoost", None),
    ("LogisticRegression", "Logistic regression", None),
    ("SVC", "Support vector machine", None),
    ("MLPClassifier", "Neural network (MLP)", None),
    ("XGBClassifier", "XGBoost", "xgboost"),
    ("CatBoostClassifier", "CatBoost", "catboost"),
    ("LGBMClassifier", "LightGBM", "lightgbm"),
]
SCALER_OPTIONS = [
    ("standard_scaler", "Standard (mean 0, standard deviation 1)", None),
    ("min_max_scaler", "Min-max (from 0 to 1)", None),
    ("robust_scaler", "Robust (median and quartiles, for outliers)", None),
]
IMPUTER_OPTIONS = [
    ("median", "Median", None),
    ("most_frequent", "Most frequent", None),
    ("knn", "K nearest neighbours", None),
    ("iterative", "Iterative", None),
]
DEFAULT_CLASSIFIERS = ["RandomForestClassifier"]
DEFAULT_SCALERS = ["standard_scaler"]
DEFAULT_IMPUTERS = ["knn"]


def package_installed(package):
    import importlib.util
    return package is None or importlib.util.find_spec(package) is not None


class OptionChecklist(QtWidgets.QWidget):
    """Checkboxes for a list of options, in a grid; options whose package is not installed are greyed out."""

    def __init__(self, options, defaults, columns=3, parent=None):
        super().__init__(parent)
        layout = QtWidgets.QGridLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setHorizontalSpacing(18)
        self.checkboxes = {}
        for index, (name, description, package) in enumerate(options):
            checkbox = QtWidgets.QCheckBox(name)
            checkbox.setToolTip(description)
            if not package_installed(package):
                checkbox.setEnabled(False)
                checkbox.setToolTip(f"{description}: needs the '{package}' package (pip install {package})")
                checkbox.setText(f"{name} (needs {package})")
            checkbox.setChecked(name in defaults and checkbox.isEnabled())
            layout.addWidget(checkbox, index // columns, index % columns)
            self.checkboxes[name] = checkbox

    def selected(self):
        return [name for name, checkbox in self.checkboxes.items() if checkbox.isChecked()]


# Each task maps the curated labels (good, MUA, noise) to the classes of one model.
# Labels missing from a mapping are left out of that model's training data.
training_tasks = {
    "noise_vs_neural": dict(
        title="Noise vs Neural (cascade layer 1)",
        suffix="_noise_vs_neural",
        mapping={"noise": "noise", "good": "neural", "MUA": "neural"},
    ),
    "sua_vs_mua": dict(
        title="SUA vs MUA (cascade layer 2, trained on neural units only)",
        suffix="_sua_vs_mua",
        mapping={"good": "good", "MUA": "MUA"},
    ),
    "sua_vs_all": dict(
        title="SUA vs all",
        suffix="_sua_vs_all",
        mapping={"good": "good", "MUA": "not-sua", "noise": "not-sua"},
    ),
}
class MetricPill(QtWidgets.QFrame):
    # Signal to emit when the X is clicked, sending the metric name back
    removed = pyqtSignal(str)

    def __init__(self, name, parent=None):
        super().__init__(parent)
        self.name = name
        
        # 1. Setup the Layout for the Pill
        layout = QtWidgets.QHBoxLayout()
        layout.setContentsMargins(5, 2, 5, 2) # Left, Top, Right, Bottom padding
        self.setLayout(layout)

        # 2. The Label (Metric Name)
        self.label = QtWidgets.QLabel(name)
        self.label.setStyleSheet("border: none; background: transparent; color: #3730A3;")
        
        # 3. The 'X' Button
        self.close_btn = QtWidgets.QPushButton("✕")
        #self.close_btn.setCursor(QtWidgets.PointingHandCursor)
        self.close_btn.setFixedSize(20, 20)
        # Styling the button to look like a clean icon
        self.close_btn.setStyleSheet("""
            QPushButton {
                border: none;
                color: #3730A3;
                font-weight: bold;
                background: transparent;
            }
            QPushButton:hover {
                color: red;
            }
        """)
        self.close_btn.clicked.connect(self.on_close_clicked)

        # 4. Add widgets to layout
        layout.addWidget(self.label)
        layout.addWidget(self.close_btn)

        # 5. Style the Pill itself (Rounded corners, background)
        self.setStyleSheet("""
            MetricPill {
                background-color: #EEF0FF;
                border-radius: 10px; 
                border: none;
            }
        """)
        
        # Make sure the pill doesn't expand indefinitely
        self.setSizePolicy(QtWidgets.QSizePolicy.Maximum, QtWidgets.QSizePolicy.Fixed)

    def on_close_clicked(self):
        # Emit the signal so the parent knows to run the specific method
        self.removed.emit(self.name)
        # Remove the visual widget
        self.close()
        self.deleteLater()

class TrainWindow(QtWidgets.QMainWindow):

    update_signal = pyqtSignal()

    def __init__(self, project):

        super().__init__()

        project_folder = project.folder_name

        model_folders_in_project = project.model_paths
        if len(model_folders_in_project) > 0:
            model_indices = np.array([int(str(model_path[0].name).split('__')[-1][0]) for model_path in model_folders_in_project if '__' in str(model_path[0].name)])
            max_model_index = np.max(model_indices) if len(model_indices) > 0 else 0
        else:
            max_model_index = 0

        self.model_name = f"model__{max_model_index + 1}"

        train_model_kwargs = {}

        window_title_text = "UNITREFINE: Train your model"
        self.setWindowTitle(window_title_text)

        self.model_folder = None

        labels = []
        metrics_paths = []
        metrics = []
        analyzers_for_training = []
        self.analyzer_folders = []
        self.analyzer_paths = []
        for analyzer in project.analyzers.values():
            analyzer_folder = project.folder_name / analyzer['analyzer_in_project']

            labels_path = analyzer_folder / "labels.csv"
            if labels_path.is_file():

                labels_df = pd.read_csv(labels_path).sort_values('unit_id')
                labels.append(labels_df['quality'].values)

                metrics_path = analyzer_folder / "labelled_metrics.csv"
                metrics_paths.append(metrics_path)
                metrics.append(pd.read_csv(metrics_path))

                analyzers_for_training.append(analyzer)
                self.analyzer_folders.append(analyzer_folder)
                self.analyzer_paths.append(str(analyzer['path']))

        train_model_kwargs['metrics_paths'] = metrics_paths
        metric_names_set = set()
        
        for metric_list in metrics:
            metric_names_for_one_sa = set(metric_list.columns)
            if len(metric_names_set) == 0:
                metric_names_set = metric_names_for_one_sa
            else:
                metric_names_set = metric_names_set.intersection(metric_names_for_one_sa)

        train_model_kwargs['labels'] = labels
        parent_folder = project_folder / 'models' 

        metric_names = [metric_name for metric_name in metric_names_set if metric_name not in ["index", "quality", "unit_id"]]        
        self.metric_names = metric_names
        self.available_metrics = copy(metric_names)

        metrics_presets["All available"] = self.available_metrics

        data_text = "Using the following analyzers:<br />"
        for analyzer_dict, metric_data in zip(analyzers_for_training, metrics, strict=True):

            data_text += f"{analyzer_dict['path']}: {len(metric_data)} units curated.<br />"

        data_text += f"<br />Metrics shared by all analyzer are: {metric_names}."

        formLayout = QtWidgets.QFormLayout()
        widget = QtWidgets.QWidget()
        widget.setObjectName("page")
        label_1 = QtWidgets.QTextEdit(f"<h3>Information</h3><p>Here, you can train many models based on the labelled data.</p>{data_text}")
        label_1.setReadOnly(True)

        blank_label = QtWidgets.QLabel("")

        self.classifier_choice = OptionChecklist(CLASSIFIER_OPTIONS, DEFAULT_CLASSIFIERS)
        self.scaler_choice = OptionChecklist(SCALER_OPTIONS, DEFAULT_SCALERS)
        self.imputer_choice = OptionChecklist(IMPUTER_OPTIONS, DEFAULT_IMPUTERS, columns=4)

        self.testSizeForm = QtWidgets.QLineEdit("0.2")

        self.modelNameForm = QtWidgets.QLineEdit(self.model_name)

        trainButton = make_primary(QtWidgets.QPushButton("Train models"))
        trainButton.clicked.connect(partial(self.do_training, train_model_kwargs, parent_folder, project))

        codeButton = QtWidgets.QPushButton("Generate code to train a model")
        codeButton.clicked.connect(partial(self.generate_code, train_model_kwargs, parent_folder))

        formLayout.addRow(label_1)
        formLayout.addRow(blank_label)
        
        row_layout = QtWidgets.QHBoxLayout()
        lbl_instruction = QtWidgets.QLabel("Which metrics?")
        lbl_instruction.setStyleSheet("font-weight: bold;")

        self.metric_selector = QtWidgets.QComboBox()
        self.metric_selector.addItems(["All available", "Spike times metrics", "Quality metrics", "Template metrics"])
        self.metric_selector.currentTextChanged.connect(self.select_metrics)
        
        row_layout.addWidget(lbl_instruction)
        row_layout.addWidget(self.metric_selector)
        row_layout.addStretch()

        formLayout.addRow(row_layout)

        # Choosing metrics
        self.scroll_area = QtWidgets.QScrollArea()
        self.scroll_area.setWidgetResizable(True)
        self.scroll_area.setFrameShape(QtWidgets.QFrame.NoFrame)
        
        self.pill_container = QtWidgets.QWidget()
        self.pill_layout = QtWidgets.QHBoxLayout(self.pill_container)
        
        self.scroll_area.setWidget(self.pill_container)
        formLayout.addRow(self.scroll_area)

        self.select_metrics('All available')

        models_label = QtWidgets.QLabel("Which models to train?")
        models_label.setStyleSheet("font-weight: bold;")
        formLayout.addRow(models_label)

        self.task_checkboxes = {}
        for task_name, task in training_tasks.items():
            checkbox = QtWidgets.QCheckBox(f"{task['title']}  ->  {self.model_name}{task['suffix']}")
            checkbox.setChecked(task_name == "sua_vs_all")
            self.task_checkboxes[task_name] = checkbox
            formLayout.addRow(checkbox)
        self.modelNameForm.textChanged.connect(self.update_task_checkbox_names)
        formLayout.addRow(blank_label)

        search_label = QtWidgets.QLabel("How to train them?")
        search_label.setStyleSheet("font-weight: bold;")
        formLayout.addRow(search_label)
        search_note = QtWidgets.QLabel(
            "The defaults work well for most recordings. If you tick more, each combination is trained and the best "
            "one is kept, which takes longer.")
        search_note.setObjectName("muted")
        search_note.setWordWrap(True)
        formLayout.addRow(search_note)
        formLayout.addRow("Classifiers: ", self.classifier_choice)
        formLayout.addRow("Scalers: ", self.scaler_choice)
        formLayout.addRow("Imputers: ", self.imputer_choice)
        formLayout.addRow(blank_label)

        formLayout.addRow("Test size: ", self.testSizeForm)
        formLayout.addRow(blank_label)

        formLayout.addRow("Model name: ", self.modelNameForm)
        formLayout.addRow(blank_label)

        formLayout.addRow(trainButton)
        formLayout.addRow(codeButton)

        widget.setLayout(formLayout)

        scroll_area = QtWidgets.QScrollArea()
        scroll_area.setWidgetResizable(True)
        scroll_area.setFrameShape(QtWidgets.QFrame.NoFrame)
        scroll_area.setWidget(widget)
        self.setCentralWidget(scroll_area)

    def update_task_checkbox_names(self, model_name):
        for task_name, checkbox in self.task_checkboxes.items():
            task = training_tasks[task_name]
            checkbox.setText(f"{task['title']}  ->  {model_name}{task['suffix']}")

    def selected_tasks(self):
        return [task_name for task_name, checkbox in self.task_checkboxes.items() if checkbox.isChecked()]

    def make_task_training_inputs(self, task_name, train_model_kwargs):
        """
        Returns the `metrics_paths` and `labels` for one training task. For tasks which use a subset
        of the labels (e.g. SUA vs MUA), the labelled metrics of the kept units are saved to
        `labelled_metrics_{task_name}.csv` in each analyzer folder.
        The units used are kept in `self.units_used`, as {analyzer path: unit ids}.
        """

        mapping = training_tasks[task_name]["mapping"]

        metrics_paths, labels = [], []
        self.units_used = {}
        for analyzer_folder, analyzer_path in zip(self.analyzer_folders, self.analyzer_paths):
            labels_df = pd.read_csv(analyzer_folder / "labels.csv").sort_values("unit_id")
            labelled_metrics = pd.read_csv(analyzer_folder / "labelled_metrics.csv", index_col=0)

            labels_df["task_label"] = labels_df["quality"].map(mapping)
            labels_df = labels_df.dropna(subset=["task_label"])
            if len(labels_df) == 0:
                continue

            task_metrics_path = analyzer_folder / f"labelled_metrics_{task_name}.csv"
            labelled_metrics.loc[labels_df["unit_id"].values].to_csv(task_metrics_path, index_label="unit_id")

            metrics_paths.append(task_metrics_path)
            labels.append(labels_df["task_label"].values)
            self.units_used[analyzer_path] = labels_df["unit_id"].values

        return metrics_paths, labels

    def generate_code(self, train_model_kwargs, parent_folder):

        code_text = "\n# Here is the code being executed by the 'Train Model' button above\n"
        code_text += "# Feel free to play with the different arguments!\n\n"
        code_text += "import spikeinterface.full as si\n\n"

        for task_name in self.selected_tasks():
            model_folder = parent_folder / f"{self.modelNameForm.text()}{training_tasks[task_name]['suffix']}"
            metrics_paths, labels = self.make_task_training_inputs(task_name, train_model_kwargs)

            code_text += f"# {training_tasks[task_name]['title']}\n"
            code_text += f"model = si.train_model(\n    mode='csv',\n    imputation_strategies={self.imputer_choice.selected()},\n    scaling_techniques={self.scaler_choice.selected()},\n    classifiers={self.classifier_choice.selected()},\n    test_size={float(self.testSizeForm.text())},\n    folder={model_folder},\n    metric_names={self.metric_names},\n"
            code_text += f"    metrics_paths = {metrics_paths},\n"
            code_text += f"    labels = {[list(label_array) for label_array in labels]},\n"
            code_text += ")\n\n"
            code_text += f"# Your model is saved at {model_folder}\n\n"

        code_text += "Read more here: https://spikeinterface.readthedocs.io/en/stable/tutorials/curation/plot_2_train_a_model.html\n\n"

        print(code_text)

    def select_metrics(self, text):

        preset_metric_names = metrics_presets[text]
        metric_names = [metric_name for metric_name in preset_metric_names if metric_name in self.available_metrics]

        self.metric_names = metric_names

        self.make_new_metrics_list(metric_names)

    def make_new_metrics_list(self, metrics):

        while self.pill_layout.count():
            item = self.pill_layout.takeAt(0)
            widget = item.widget()
            
            if widget is not None:
                widget.deleteLater()

        for metric in metrics:

            pill = MetricPill(metric)
            pill.removed.connect(self.remove_metric_from_list)
            self.pill_layout.addWidget(pill)


    def remove_metric_from_list(self, metric_name):
        self.metric_names = [existing_metric_name for existing_metric_name in self.metric_names if existing_metric_name != metric_name]

    def do_training(self, train_model_kwargs, parent_folder, project):

        imputation_strategies = self.imputer_choice.selected()
        scaling_techniques = self.scaler_choice.selected()
        classifiers = self.classifier_choice.selected()
        for name, chosen in [("classifier", classifiers), ("scaler", scaling_techniques), ("imputer", imputation_strategies)]:
            if len(chosen) == 0:
                QtWidgets.QMessageBox.information(self, "Choose how to train", f"Tick at least one {name}.")
                return
        try:
            test_size = float(self.testSizeForm.text())
            assert 0 < test_size < 1
        except (ValueError, AssertionError):
            QtWidgets.QMessageBox.information(self, "Test size", "The test size must be a number between 0 and 1, e.g. 0.2.")
            return
        self.model_name = self.modelNameForm.text()

        task_names = self.selected_tasks()
        if len(task_names) == 0:
            print("No models selected - tick at least one model type to train.")
            return

        folders = {task_name: parent_folder / f"{self.model_name}{training_tasks[task_name]['suffix']}" for task_name in task_names}
        existing_folders = [str(folder) for folder in folders.values() if folder.is_dir()]
        if len(existing_folders) > 0:
            print(f"Model folders already exist: {existing_folders}. Please choose another model name.")
            return

        print("\nUsing UnitRefine, inside SpikeInterface, to train models...\n")

        notice = BusyNotice(self, "Training models", "").__enter__()

        trained_folders = []
        try:
            for task_index, task_name in enumerate(task_names):
                notice.update(
                    f"{task_index + 1} of {len(task_names)}: {training_tasks[task_name]['title']}. This can take a few "
                    "minutes; the progress is printed in the terminal."
                )

                folder = folders[task_name]
                metrics_paths, labels = self.make_task_training_inputs(task_name, train_model_kwargs)

                classes = set(np.concatenate(labels)) if len(labels) > 0 else set()
                if len(classes) < 2:
                    print(f"\nSkipping '{training_tasks[task_name]['title']}': it needs units from at least two classes, found {sorted(classes)}.")
                    continue

                print(f"\n=== Training '{training_tasks[task_name]['title']}' on {sum(len(label_array) for label_array in labels)} units ===\n")

                # an explicit seed, so the test split can be made again for the figures
                seed = int(np.random.default_rng().integers(0, 2**31))
                train_model(
                    mode="csv",
                    imputation_strategies=imputation_strategies,
                    scaling_techniques=scaling_techniques,
                    classifiers=classifiers,
                    test_size=test_size,
                    folder=folder,
                    metric_names=self.metric_names,
                    metrics_paths=metrics_paths,
                    labels=labels,
                    seed=seed,
                )

                balanced_accuracy, precision, recall = pd.read_csv(folder / 'model_accuracies.csv')[['balanced_accuracy', 'precision', 'recall']].values[0]
                print(f"\nBest model stats:\n    Balanced Accuray: {balanced_accuracy}\n    Precision: {precision}\n    Recall: {recall}")

                # record which human labels the model learned from, so the figures can leave them out
                write_trained_on(folder, self.units_used)
                try:
                    save_test_predictions(folder, test_size, seed)
                except Exception as e:
                    print(f"Could not save the test set predictions of {folder.name}: {e}")

                project.model_paths.append((folder, "local"))
                trained_folders.append(folder)

        finally:
            notice.__exit__(None, None, None)

        if len(trained_folders) == 0:
            print("\nNo models were trained.\n")
            return

        print("\nFinished training models! Best models saved in:")
        for folder in trained_folders:
            print(f"    {folder}")
        print()

        self.model_folder = trained_folders[-1]

        self.close()


    def closeEvent(self, event):
        """Intercepts the user closing the app, to save the labels."""

        self.update_signal.emit()
        #self.parent_window.update_signal.emit()
        event.accept()
