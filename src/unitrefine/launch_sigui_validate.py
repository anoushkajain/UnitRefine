import argparse
import functools
import sys
from pathlib import Path

import pandas as pd
import spikeinterface.full as si
from PyQt5.QtGui import QCloseEvent
from PyQt5.QtWidgets import QWidget
from pyqtgraph import mkQApp
from spikeinterface_gui.backend_qt import QtMainWindow
from spikeinterface_gui.controller import Controller
from spikeinterface.curation import auto_label_units
import numpy as np

import warnings
warnings.filterwarnings("ignore")

def my_custom_close_handler(event: QCloseEvent, window: QWidget, project_folder, save_folder, analyzer, model_name):
    """
    This function will be called instead of the original closeEvent.
    """

    re_labelled_unit_ids = []
    re_labels = []

    for row in curation_dict['manual_labels']:
        
        if row.get('labels') is None:
            if row.get('quality') is not None:
                re_labels.append(row['quality'][0])
                re_labelled_unit_ids.append(int(row['unit_id']))
        else:
            if row.get('labels').get('quality') is not None:
                re_labels.append(row['labels']['quality'][0])
                re_labelled_unit_ids.append(int(row['unit_id']))

    labels_df = pd.DataFrame()
    labels_df['quality'] = re_labels
    labels_df['unit_id'] = re_labelled_unit_ids
    labels_df = labels_df.sort_values('unit_id')

    relabelled_units_path = save_folder / f"relabelled_units_{model_name}.csv"

    labels_df.to_csv(relabelled_units_path, index=False)

    all_metrics_path = save_folder / "all_metrics.csv"
    if not all_metrics_path.is_file():
        if analyzer.has_extension('template_metrics'):
            tms = analyzer.get_extension("template_metrics").get_data()
        else:
            tms = pd.DataFrame()
        
        if analyzer.has_extension('quality_metrics'):
            qms = analyzer.get_extension("quality_metrics").get_data()
        else:
            qms = pd.DataFrame()
        
        all_metrics = pd.concat([qms, tms], axis=1)
        all_metrics.to_csv(all_metrics_path, index_label="unit_id")

argv = sys.argv[1:]

parser = argparse.ArgumentParser(description='spikeinterface-gui')
parser.add_argument('analyzer_folder', help='SortingAnalyzer folder path', default=None, nargs='?')
parser.add_argument('project_folder', help='Project folder path', default=None, nargs='?')
parser.add_argument('analyzer_in_project', help='Project folder path', default=None, nargs='?')
parser.add_argument('analyzer_index', help='Project folder path', default=None, nargs='?')
parser.add_argument('model_folder')
parser.add_argument('current_model_name')
parser.add_argument('hfh_or_local')
parser.add_argument('relabel')
parser.add_argument('percentage_float')


args = parser.parse_args(argv)

project_folder = Path(args.project_folder)
analyzer_in_project = Path(args.analyzer_in_project)
analyzer_index = args.analyzer_index
analyzer_folder = args.analyzer_folder
model_folder = args.model_folder
current_model_name = args.current_model_name
hfh_or_local = args.hfh_or_local
relabel = True if args.relabel == "True" else False 
percentage_float = float(args.percentage_float)

if '//' not in analyzer_folder:
    analyzer_folder = Path(analyzer_folder)

save_folder = project_folder / analyzer_in_project
save_folder.mkdir(exist_ok=True, parents=True)

sorting_analyzer = si.load_sorting_analyzer(analyzer_folder)#, load_extensions=False)

print("\nUsing UnitRefine to compute predictions made by your model.\n")

model_decisions = auto_label_units(sorting_analyzer=sorting_analyzer, model_folder=model_folder, trust_model=True)

model_labels_filepath = f"{project_folder / analyzer_in_project / f'labels_from_{current_model_name}.csv'}"
model_decisions.to_csv(model_labels_filepath, index_label="unit_id")
model_decisions['unit_id'] = model_decisions.index

manual_labels = []
for unit_id in sorting_analyzer.unit_ids:
    decision = {"unit_id": unit_id, 
                "model": model_decisions[model_decisions['unit_id'] == unit_id]['prediction'].values,
                }
    if len(decision['model']) > 0:
        manual_labels.append(decision)
    

label_definitions = {
    "quality": dict(name="quality", label_options=["good", "MUA", "noise"], exclusive=True),
    "model": dict(name="model", label_options=["good", "MUA", "noise"], exclusive=True),
}

extra_unit_properties = {'confidence': model_decisions['probability'].values}

if relabel:
    
    probability = model_decisions['probability'].values
    num_units_to_keep = int(len(probability) * percentage_float/100)
    least_confident_indices = np.argpartition(probability, num_units_to_keep)[:num_units_to_keep]
    least_confident_unit_ids = sorting_analyzer.unit_ids[least_confident_indices]
    sorting_analyzer = sorting_analyzer.select_units(unit_ids=least_confident_unit_ids)
    extra_unit_properties['confidence'] = probability[least_confident_indices]
    
    manual_labels = []
    for unit_id in sorting_analyzer.unit_ids:
        decision = {"unit_id": unit_id, 
                    "model": model_decisions[model_decisions['unit_id'] == unit_id]['prediction'].values,
                    }
        if len(decision['model']) > 0:
            manual_labels.append(decision)
    

curation_dict = dict(
    format_version="2",
    unit_ids=sorting_analyzer.unit_ids,
    manual_labels=manual_labels,
    label_definitions=label_definitions,
)

controller = Controller(
    sorting_analyzer,
    backend="qt",
    curation=True,
    curation_data=curation_dict,
    extra_unit_properties=extra_unit_properties,
    displayed_unit_properties=['model', 'quality', 'confidence', 'firing_rate', 'snr', 'x', 'y', 'rp_violations'],
    skip_extensions=['waveforms', 'principal_components', 'spike_locations', 'isi_histograms', 'template_similarity'],
)

layout_dict={'zone1': ['unitlist', 'mainsettings'], 'zone2': [], 'zone3': ['waveform'], 'zone4': ['correlogram'], 'zone5': ['spikeamplitude'], 'zone6': [], 'zone7': [], 'zone8': ['spikerate']}

if relabel:
    print(f"Launching SpikeInterface-GUI to relabel automated curation for analyzer at {analyzer_folder}")
else:
    print(f"Launching SpikeInterface-GUI to inspect model predictions for analyzer at {analyzer_folder}")

print("Main UnitRefine window will remain inactive until you close the SpikeInterface-GUI window")
print("Re-label units as noise, good and MUA by pressing 'n', 'g' and 'm' on your keyboard.")

model_name = Path(model_folder).name

app = mkQApp()
win = QtMainWindow(controller, layout_dict=layout_dict, user_settings=None)
win.closeEvent = functools.partial(my_custom_close_handler, window=win, project_folder=project_folder, save_folder=save_folder, analyzer=sorting_analyzer, model_name=model_name)

win.show()
app.exec()
