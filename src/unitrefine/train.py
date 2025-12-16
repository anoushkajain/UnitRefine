from datetime import datetime
from functools import partial

import pandas as pd
import PyQt5.QtWidgets as QtWidgets
from PyQt5.QtCore import pyqtSignal
from PyQt5.QtCore import Qt
from spikeinterface.curation import train_model
import numpy as np
from copy import copy

metrics_presets = {
    "Spike times metrics": ["num_spikes", "firing_rate", "presence_ratio", "snr", "isi_violations_ratio", "isi_violation_count", "rp_contamination", "rp_violations", "sliding_rp_violation", "sync_spike_2", "sync_spike_4", "sync_spike_8", "firing_range"],
    "Quality metrics": ["num_spikes", "firing_rate", "presence_ratio", "snr", "isi_violations_ratio", "isi_violation_count", "rp_contamination", "rp_violations", "sliding_rp_violation", "sync_spike_2", "sync_spike_4", "sync_spike_8", "firing_range", "amplitude_cutoff", "amplitude_median", "amplitude_cv_median", "amplitude_cv_range", "drift_ptp", "drift_std", "drift_mad", "isolation_distance", "l_ratio", "d_prime", "silhouette", "nn_hit_rate", "nn_miss_rate"],
    "Template metrics": ["peak_to_valley", "peak_trough_ratio", "half_width", "repolarization_slope", "recovery_slope", "num_positive_peaks", "num_negative_peaks", "velocity_above" "velocity_below", "exp_decay", "spread"],
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
        self.label.setStyleSheet("border: none; background-color: white; font-weight: bold;")
        
        # 3. The 'X' Button
        self.close_btn = QtWidgets.QPushButton("✕")
        #self.close_btn.setCursor(QtWidgets.PointingHandCursor)
        self.close_btn.setFixedSize(20, 20)
        # Styling the button to look like a clean icon
        self.close_btn.setStyleSheet("""
            QPushButton {
                border: none;
                color: black;
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
                background-color: white;
                border-radius: 10px; 
                border: 1px solid #000000;
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
        widget.setStyleSheet("background-color: PeachPuff")
        label_1 = QtWidgets.QTextEdit(f"<h3>Information</h3><p>Here, you can train many models based on the labelled data.</p>{data_text}")
        label_1.setReadOnly(True)
        label_1.setStyleSheet("background-color: white")

        blank_label = QtWidgets.QLabel("")

        self.classifiersForm = QtWidgets.QLineEdit("['RandomForestClassifier']")
        self.classifiersForm.setStyleSheet("background-color: white")
        classifiersOptions = QtWidgets.QLabel("<i>Possible options</i>: 'RandomForestClassifier', 'AdaBoostClassifier', 'GradientBoostingClassifier', 'SVC',<br /> 'LogisticRegression', 'XGBClassifier', 'CatBoostClassifier', 'LGBMClassifier', 'MLPClassifier'.")

        self.scalarsForm = QtWidgets.QLineEdit("['standard_scaler']")
        self.scalarsForm.setStyleSheet("background-color: white")

        scalarsOptions = QtWidgets.QLabel("<i>Possible options</i>: 'standard_scaler', 'min_max_scaler', 'robust_scaler'")

        self.imputersForm = QtWidgets.QLineEdit("['knn']")
        self.imputersForm.setStyleSheet("background-color: white")
        imputersOptions = QtWidgets.QLabel("<i>Possible options</i>: 'median', 'most_frequent', 'knn', 'iterative'")

        self.testSizeForm = QtWidgets.QLineEdit("0.2")
        self.testSizeForm.setStyleSheet("background-color: white")

        self.modelNameForm = QtWidgets.QLineEdit(self.model_name)
        self.modelNameForm.setStyleSheet("background-color: white")

        trainButton = QtWidgets.QPushButton("Train models")
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

        formLayout.addRow("List of Classifiers: ", self.classifiersForm)
        formLayout.addRow(classifiersOptions)
        formLayout.addRow(blank_label)

        formLayout.addRow("List of Scalars: ", self.scalarsForm)
        formLayout.addRow(scalarsOptions)
        formLayout.addRow(blank_label)

        formLayout.addRow("List of Imputers: ", self.imputersForm)
        formLayout.addRow(imputersOptions)
        formLayout.addRow(blank_label)

        formLayout.addRow("Test size: ", self.testSizeForm)
        formLayout.addRow(blank_label)

        formLayout.addRow("Model name: ", self.modelNameForm)
        formLayout.addRow(blank_label)

        formLayout.addRow(trainButton)
        formLayout.addRow(codeButton)

        widget.setLayout(formLayout)

        self.setCentralWidget(widget)

    def generate_code(self, train_model_kwargs, parent_folder):

        model_folder = parent_folder / f'model_{datetime.now():%Y-%m-%d_%H:%M:%S}'

        code_text = "\n# Here is the code being executed by the 'Train Model' button above\n"
        code_text += "# Feel free to play with the different arguments!\n\n"
        code_text += "import spikeinterface.full as si\n\n"
        code_text += f"model = si.train_model(\n    mode='csv',\n     imputation_strategies={eval(self.imputersForm.text())},\n    scaling_techniques={eval(self.scalarsForm.text())},\n    classifiers={eval(self.classifiersForm.text())},\n    test_size={eval(self.testSizeForm.text())},\n    folder={model_folder}\n"
        for key, value in train_model_kwargs.items():
            code_text += f"    {key} = {value},\n"
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
                # deleteLater schedules the widget for deletion
                widget.deleteLater()

        for metric in metrics:

            # Create the pill
            pill = MetricPill(metric)
            # Connect the custom signal to your specific removal method
            pill.removed.connect(self.remove_metric_from_list)
            
            # Add to layout
            self.pill_layout.addWidget(pill)
            #self.input_field.clear()

    # --- This is the specific method you requested ---
    def remove_metric_from_list(self, metric_name):
        self.metric_names = [existing_metric_name for existing_metric_name in self.metric_names if existing_metric_name != metric_name]

    def do_training(self, train_model_kwargs, parent_folder, project):

        imputation_strategies = eval(self.imputersForm.text())
        scaling_techniques = eval(self.scalarsForm.text())
        classifiers = eval(self.classifiersForm.text())
        test_size = eval(self.testSizeForm.text())
        self.model_name = self.modelNameForm.text()

        folder = parent_folder / self.model_name

        print("\nUsing UnitRefine, inside SpikeInterface, to train models...\n")

        QtWidgets.QApplication.setOverrideCursor(Qt.CursorShape.WaitCursor)

        try:
          
            train_model(
                mode="csv",
                imputation_strategies=imputation_strategies,
                scaling_techniques=scaling_techniques,
                classifiers=classifiers,
                test_size=test_size,
                folder=folder,
                metric_names=self.metric_names,
                **train_model_kwargs,
            )

            balanced_accuracy, precision, recall = pd.read_csv(folder / 'model_accuracies.csv')[['balanced_accuracy', 'precision', 'recall']].values[0]
            print(f"\nBest model stats:\n    Balanced Accuray: {balanced_accuracy}\n    Precision: {precision}\n    Recall: {recall}")

        finally:
            QtWidgets.QApplication.restoreOverrideCursor()

        project.model_paths.append((folder, "local"))

        print(f"\nFinished training models! Best model saved in in {folder}.\n")

        self.model_folder = folder

        self.close()


    def closeEvent(self, event):
        """Intercepts the user closing the app, to save the labels."""

        self.update_signal.emit()
        #self.parent_window.update_signal.emit()
        event.accept()
