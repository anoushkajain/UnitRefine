"""
Runs one of UnitRefine's SpikeInterface-GUI launch scripts without a person: once its window is open, sets
labels the way the n, g and m keys do (controller.set_label_to_unit), then closes the window, so the
script saves the labels as it does for a user. Usage:

    python sigui_harness.py <launch script> <its arguments...> --labels unit_id=label,unit_id=label
"""

import runpy
import sys

import pyqtgraph
import PyQt5.QtWidgets as QtWidgets

labels_argument = sys.argv[sys.argv.index("--labels") + 1]
labels = dict(item.split("=") for item in labels_argument.split(",")) if labels_argument else {}
script = sys.argv[1]
sys.argv = [script] + sys.argv[2:sys.argv.index("--labels")]

original_mkQApp = pyqtgraph.mkQApp


def mkQApp(*args, **kwargs):
    app = original_mkQApp(*args, **kwargs)

    def label_and_close():
        window = [widget for widget in QtWidgets.QApplication.topLevelWidgets() if hasattr(widget, "controller")][0]
        controller = window.controller
        unit_ids = {str(unit_id): unit_id for unit_id in controller.unit_ids}
        for unit_id, label in labels.items():
            controller.set_label_to_unit(unit_ids[unit_id], "quality", label)
        print("SHOWN_UNITS", ",".join(unit_ids), flush=True)
        window.close()
        return 0

    app.exec = label_and_close
    app.exec_ = label_and_close
    return app


# the launch scripts import mkQApp from pyqtgraph when they run, so they get this one
pyqtgraph.mkQApp = mkQApp
runpy.run_path(script, run_name="__main__")
