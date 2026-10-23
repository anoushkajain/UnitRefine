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
    analyzer_unit_ids = {str(unit_id): unit_id for unit_id in analyzer.unit_ids}

    # the labels as they are now: spikeinterface-gui >= 0.13 works on its own copy of the curation data,
    # so the dictionary it was given does not change
    for row in window.controller.curation_data['manual_labels']:
        quality = row.get('quality') if row.get('labels') is None else row['labels'].get('quality')
        if quality is None or len(quality) == 0:
            continue
        labelled_unit_ids.append(analyzer_unit_ids[str(row['unit_id'])])
        labels.append(quality[0])

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

report_sigui_step("loading")
analyzer = si.load_sorting_analyzer(str(analyzer_folder), load_extensions=False)

# the existing labels fill the quality column. Unit ids are matched as text: labels.csv is read back with
# pandas' inferred type (e.g. numbers), while an analyzer's unit ids can be text
existing_labels = dict(zip(decisions['unit_id'].astype(str), decisions['quality']))
manual_labels = []
for unit_id in analyzer.unit_ids:
    quality = existing_labels.get(str(unit_id))
    if isinstance(quality, str):
        # curation format 2: the labels of a unit are under "labels"
        manual_labels.append({"unit_id": unit_id, "labels": {"quality": [quality]}})

label_definitions = {
    "quality": dict(name="quality", label_options=["good", "MUA", "noise"], exclusive=True),
}

curation_dict = dict(
    format_version="2",
    unit_ids=analyzer.unit_ids,
    manual_labels=manual_labels,
    label_definitions=label_definitions,
)

report_sigui_step("views")
controller = Controller(
        analyzer, backend="qt", curation=True, curation_data=curation_dict, skip_extensions=['waveforms', 'principal_components', 'spike_locations', 'isi_histograms', 'template_similarity', 'amplitudescalings'],
)

layout_dict={'zone1': ['unitlist', 'mainsettings'], 'zone2': [], 'zone3': ['waveform'], 'zone4': ['correlogram'], 'zone5': ['spikeamplitude'], 'zone6': [], 'zone7': [], 'zone8': ['spikerate']}


app = mkQApp()
win = QtMainWindow(controller, layout_dict=layout_dict, user_settings=None)
win.closeEvent = functools.partial(my_custom_close_handler, window=win, save_folder=save_folder, analyzer=analyzer)

win.show()
# tell the UnitRefine window that the SpikeInterface-GUI is open, once the event loop has started
QTimer.singleShot(0, lambda: print(SIGUI_SHOWN_MARKER, flush=True))
app.exec()




