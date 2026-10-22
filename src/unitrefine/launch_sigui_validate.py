import argparse
import functools
import sys
from pathlib import Path

import pandas as pd
import spikeinterface.full as si
from PyQt5.QtGui import QCloseEvent
from PyQt5.QtWidgets import QWidget
from pyqtgraph import mkQApp
from PyQt5.QtCore import QTimer
from unitrefine import SIGUI_SHOWN_MARKER, check_sigui_version, report_sigui_step

check_sigui_version()
from unitrefine.import_labels import compute_all_metrics
from spikeinterface_gui.backend_qt import QtMainWindow
from spikeinterface_gui.controller import Controller
import numpy as np

from unitrefine.predictions import (
    get_saved_or_new_predictions,
    predictions_table,
    predictions_file_name,
    describe_prediction_error,
    save_csv_atomic,
)

import warnings
warnings.filterwarnings("ignore")

# extensions the SpikeInterface-GUI does not need here, and which can be large
SKIPPED_EXTENSIONS = ['waveforms', 'principal_components', 'spike_locations', 'isi_histograms', 'template_similarity', 'amplitudescalings']


def get_quality_labels(curation_dict):
    """The quality labels set in the SpikeInterface-GUI, as a Series indexed by unit id."""
    quality_labels = {}
    for row in curation_dict['manual_labels']:
        if row.get('labels') is None:
            quality = row.get('quality')
        else:
            quality = row['labels'].get('quality')
        if quality is not None and len(quality) > 0:
            quality_labels[row['unit_id']] = quality[0]
    return pd.Series(quality_labels, dtype=object)


def my_custom_close_handler(event: QCloseEvent, window: QWidget, save_folder, file_prefix, analyzer, selection_key, shown_unit_ids):
    """
    This function will be called instead of the original closeEvent.
    """

    # the labels as they are now: spikeinterface-gui >= 0.13 works on its own copy of the curation data,
    # so the dictionary it was given does not change
    quality_labels = get_quality_labels(window.controller.curation_data)

    # the quality column starts with the human labels the models were trained on; only labels which differ
    # from them (or units which had none) are new, so only those are saved as relabelled units
    new_labels = quality_labels[[original_quality.get(unit_id) != quality for unit_id, quality in quality_labels.items()]]

    labels_df = pd.DataFrame({'quality': new_labels.values, 'unit_id': new_labels.index})

    # keep the labels from earlier sessions for units which were not shown this time
    if existing_relabels is not None:
        shown = set(str(unit_id) for unit_id in shown_unit_ids)
        kept = existing_relabels[~existing_relabels['unit_id'].astype(str).isin(shown)]
        labels_df = pd.concat([kept[['quality', 'unit_id']], labels_df])

    labels_df = labels_df.sort_values('unit_id', key=lambda unit_ids: unit_ids.astype(str))
    save_csv_atomic(labels_df, relabelled_units_path, index=False)

    # the metrics of all units, which Retrain uses; `analyzer` is the full analyzer, not only the units shown
    all_metrics_path = save_folder / f"{file_prefix}all_metrics.csv"
    if not all_metrics_path.is_file():
        compute_all_metrics(analyzer).to_csv(all_metrics_path, index_label="unit_id")

argv = sys.argv[1:]

parser = argparse.ArgumentParser(description='spikeinterface-gui')
parser.add_argument('analyzer_folder', help='SortingAnalyzer folder path')
parser.add_argument('--save_folder', required=True, help='Folder for the predictions and relabelled units')
parser.add_argument('--file_prefix', default='', help='Prefix for the saved file names')
parser.add_argument('--mode', choices=['binary', 'cascade'], default='binary')
parser.add_argument('--model_folders', nargs='+', required=True, help='Binary: the models. Cascade: layer 1 then layer 2.')
parser.add_argument('--selection_key', required=True, help='Name used for the relabelled units file')
parser.add_argument('--human_labels', default=None, help='csv with the curated labels (unit_id, quality), which fill the quality column')
parser.add_argument('--relabel', action='store_true', help='Only show the units in --units_file')
parser.add_argument('--units_file', default=None, help='csv with the unit_id of the units to relabel')


args = parser.parse_args(argv)

save_folder = Path(args.save_folder)
file_prefix = args.file_prefix
analyzer_folder = args.analyzer_folder
mode = args.mode
model_folders = [Path(model_folder) for model_folder in args.model_folders]
selection_key = args.selection_key
relabel = args.relabel

if '//' not in analyzer_folder:
    analyzer_folder = Path(analyzer_folder)

save_folder.mkdir(exist_ok=True, parents=True)

# extensions are loaded when they are needed, so large ones like waveforms or principal components are not read
report_sigui_step("loading")
sorting_analyzer = si.load_sorting_analyzer(analyzer_folder, load_extensions=False)
# kept for saving the metrics of all units, also when only some units are shown for relabelling
full_sorting_analyzer = sorting_analyzer
# unit ids read back from csv files get pandas' inferred dtype, so match them as strings
analyzer_unit_ids = {str(unit_id): unit_id for unit_id in sorting_analyzer.unit_ids}

if mode == "cascade":
    print(f"\nPredictions of the cascade: {model_folders[0].name} (layer 1), then {model_folders[1].name} (layer 2) on the neural units.\n")
else:
    print(f"\nPredictions made by: {[model_folder.name for model_folder in model_folders]}.\n")

report_sigui_step("predictions")
predictions_path = save_folder / predictions_file_name(file_prefix, selection_key)
try:
    # reuse the predictions saved by an earlier Inspect, Relabel or Apply, when they are up to date
    label_columns, confidence_columns, selection_confidence = get_saved_or_new_predictions(
        sorting_analyzer, analyzer_folder, mode, model_folders, predictions_path
    )
except Exception as e:
    print(f"\nCould not compute predictions for {analyzer_folder}: {describe_prediction_error(e)}\n")
    sys.exit(1)


# the human labels the models were trained on, for analyzers which were curated or imported. As in Curate,
# they fill the quality column, where the n, g and m keyboard shortcuts set labels
original_quality = {}
if args.human_labels is not None and Path(args.human_labels).is_file():
    human_labels_df = pd.read_csv(args.human_labels)
    original_quality = {
        analyzer_unit_ids[str(unit_id)]: quality
        for unit_id, quality in zip(human_labels_df['unit_id'], human_labels_df['quality'])
        if str(unit_id) in analyzer_unit_ids
    }

# save every model's predictions, named after the model they come from, with the human labels next to them
predictions_df = predictions_table(
    sorting_analyzer.unit_ids, {"human_label": pd.Series(original_quality, dtype=object), **label_columns}, confidence_columns
)
save_csv_atomic(predictions_df, predictions_path)

# labels from earlier sessions with this model selection are shown again, and kept when saving
relabelled_units_path = save_folder / f"{file_prefix}relabelled_units_{selection_key}.csv"
existing_relabels = pd.read_csv(relabelled_units_path) if relabelled_units_path.is_file() else None
existing_quality = {}
if existing_relabels is not None:
    for unit_id, quality in zip(existing_relabels['unit_id'], existing_relabels['quality']):
        if str(unit_id) in analyzer_unit_ids:
            existing_quality[analyzer_unit_ids[str(unit_id)]] = quality

unit_ids_to_show = list(sorting_analyzer.unit_ids)
if relabel:
    units_to_relabel = pd.read_csv(args.units_file)['unit_id'].astype(str)
    unit_ids_to_show = [analyzer_unit_ids[unit_id] for unit_id in units_to_relabel if unit_id in analyzer_unit_ids]
    # the selected units are copied into a new analyzer in memory, which only has the extensions loaded now,
    # so load the ones the SpikeInterface-GUI shows (not the large ones it skips)
    for extension_name in sorting_analyzer.get_saved_extension_names():
        if extension_name not in SKIPPED_EXTENSIONS:
            sorting_analyzer.load_extension(extension_name)
    sorting_analyzer = sorting_analyzer.select_units(unit_ids=unit_ids_to_show)

manual_labels = []
for unit_id in unit_ids_to_show:
    unit_labels = {}
    # labels from earlier sessions take priority over the original human labels. When relabelling, the
    # original labels are not shown, so the uncertain units get a fresh look without being nudged
    if unit_id in existing_quality:
        unit_labels["quality"] = [existing_quality[unit_id]]
    elif unit_id in original_quality and not relabel:
        unit_labels["quality"] = [original_quality[unit_id]]
    for column_name, labels in label_columns.items():
        if unit_id in labels.index:
            unit_labels[column_name] = [labels[unit_id]]
    # curation format 2: the labels of a unit are under "labels". In this format the n/g/m keys can set the
    # quality label of every unit, also of units shown with a model's label but no quality label yet
    manual_labels.append({"unit_id": unit_id, "labels": unit_labels})

label_definitions = {
    "quality": dict(name="quality", label_options=["good", "MUA", "noise"], exclusive=True),
}
for column_name, labels in label_columns.items():
    label_definitions[column_name] = dict(name=column_name, label_options=sorted(labels.unique()), exclusive=True)

unit_positions = {unit_id: position for position, unit_id in enumerate(predictions_df.index)}
shown_positions = [unit_positions[unit_id] for unit_id in unit_ids_to_show]
extra_unit_properties = {column_name: np.asarray(confidences)[shown_positions] for column_name, confidences in confidence_columns.items()}

curation_dict = dict(
    format_version="2",
    unit_ids=sorting_analyzer.unit_ids,
    manual_labels=manual_labels,
    label_definitions=label_definitions,
)

report_sigui_step("views")
controller = Controller(
    sorting_analyzer,
    backend="qt",
    curation=True,
    curation_data=curation_dict,
    extra_unit_properties=extra_unit_properties,
    # the label columns are shown by the curation view, and the confidence columns are added from extra_unit_properties
    displayed_unit_properties=['firing_rate', 'snr', 'x', 'y', 'rp_violations'],
    skip_extensions=SKIPPED_EXTENSIONS,
)

layout_dict={'zone1': ['unitlist', 'mainsettings'], 'zone2': [], 'zone3': ['waveform'], 'zone4': ['correlogram'], 'zone5': ['spikeamplitude'], 'zone6': [], 'zone7': [], 'zone8': ['spikerate']}

if relabel:
    print(f"Launching SpikeInterface-GUI to relabel the {len(unit_ids_to_show)} least confident units of analyzer at {analyzer_folder}")
else:
    print(f"Launching SpikeInterface-GUI to inspect model predictions for analyzer at {analyzer_folder}")

print("Main UnitRefine window will remain inactive until you close the SpikeInterface-GUI window")
print("Label units as noise, good and MUA by pressing 'n', 'g' and 'm' on your keyboard.")
if relabel:
    print("Your labels are shown in the 'quality' column, which starts empty (except units you relabelled before).")
else:
    print("Your labels are shown in the 'quality' column, which starts with your existing human labels.")
print("To change a label, select the unit and press another key: the new label replaces the old one.")
print("Your labels are saved automatically when you close the SpikeInterface-GUI.")

app = mkQApp()
win = QtMainWindow(controller, layout_dict=layout_dict, user_settings=None)
win.closeEvent = functools.partial(my_custom_close_handler, window=win, save_folder=save_folder, file_prefix=file_prefix, analyzer=full_sorting_analyzer, selection_key=selection_key, shown_unit_ids=unit_ids_to_show)

win.show()
# tell the UnitRefine window that the SpikeInterface-GUI is open, once the event loop has started
QTimer.singleShot(0, lambda: print(SIGUI_SHOWN_MARKER, flush=True))
app.exec()
