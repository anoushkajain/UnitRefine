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

import warnings
warnings.filterwarnings("ignore")


def my_custom_close_handler(event: QCloseEvent, window: QWidget, save_folder, analyzer):
    """
    This function will be called instead of the original closeEvent.
    """

    labelled_unit_ids = []
    labels = []

    for row in curation_dict['manual_labels']:
        labelled_unit_ids.append(int(row['unit_id']))
        if row.get('labels') is None:
            labels.append(row['quality'][0])
        else:
            labels.append(row['labels']['quality'][0])

    if analyzer.has_extension('template_metrics'):
        tms = analyzer.get_extension("template_metrics").get_data()
    else:
        tms = pd.DataFrame()
    
    if analyzer.has_extension('quality_metrics'):
        qms = analyzer.get_extension("quality_metrics").get_data()
    else:
        qms = pd.DataFrame()
    
    all_metrics = pd.concat([qms, tms], axis=1)

    labels_df = pd.DataFrame()
    labels_df['quality'] = labels
    labels_df['unit_id'] = labelled_unit_ids
    labels_df = labels_df.sort_values('unit_id')

    labelled_metrics = all_metrics[all_metrics.index.isin(labelled_unit_ids)]

    all_metrics.to_csv(save_folder / "all_metrics.csv", index_label="unit_id")
    labelled_metrics.to_csv(save_folder / "labelled_metrics.csv", index_label="unit_id")
    labels_df.to_csv(save_folder / "labels.csv", index=False)


argv = sys.argv[1:]

parser = argparse.ArgumentParser(description='spikeinterface-gui')
parser.add_argument('analyzer_folder', help='SortingAnalyzer folder path', default=None, nargs='?')
parser.add_argument('project_folder', help='Project folder path', default=None, nargs='?')
parser.add_argument('analyzer_index', help='Project folder path', default=None, nargs='?')

args = parser.parse_args(argv)


analyzer_folder = args.analyzer_folder

if '//' not in analyzer_folder:
    analyzer_folder = Path(analyzer_folder)

project_folder = Path(args.project_folder)
analyzer_index = int(args.analyzer_index)

save_folder = project_folder / (f"analyzers/analyzer_{analyzer_index}")

if Path(save_folder / "labels.csv").is_file():
    decisions = pd.read_csv(save_folder / "labels.csv")
else:
    decisions = pd.DataFrame(columns=['unit_id', 'quality'])

analyzer = si.load_sorting_analyzer(str(analyzer_folder), load_extensions=False)

manual_labels = []
for unit_id in analyzer.unit_ids:
    decision = {"unit_id": unit_id, "quality": decisions[decisions['unit_id'] == unit_id]['quality'].values}
    if len(decision['quality']) > 0:
        manual_labels.append(decision)

label_definitions = {
    "quality": dict(name="quality", label_options=["good", "MUA", "noise"], exclusive=True),
}

curation_dict = dict(
    format_version="2",
    unit_ids=analyzer.unit_ids,
    manual_labels=manual_labels,
    label_definitions=label_definitions,
)

controller = Controller(
        analyzer, backend="qt", curation=True, curation_data=curation_dict, skip_extensions=['waveforms', 'principal_components', 'spike_locations', 'isi_histograms', 'template_similarity'],
)

layout_dict={'zone1': ['unitlist', 'mainsettings'], 'zone2': [], 'zone3': ['waveform'], 'zone4': ['correlogram'], 'zone5': ['spikeamplitude'], 'zone6': [], 'zone7': [], 'zone8': ['spikerate']}


app = mkQApp()
win = QtMainWindow(controller, layout_dict=layout_dict, user_settings=None)
win.closeEvent = functools.partial(my_custom_close_handler, window=win, save_folder=save_folder, analyzer=analyzer)

win.show()
app.exec()




