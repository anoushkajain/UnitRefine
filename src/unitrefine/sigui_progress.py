import math
import time

import PyQt5.QtWidgets as QtWidgets
from PyQt5.QtCore import Qt, QTimer, pyqtSignal

from unitrefine.widgets import sigui_help_box

# The steps the SpikeInterface-GUI scripts go through while opening, as (step name, text shown).
# "start" lasts from starting the process until the script reports its first step.
CURATE_STEPS = [
    ("start", "Starting Python and SpikeInterface"),
    ("loading", "Loading the recording"),
    ("views", "Preparing the SpikeInterface-GUI views"),
]
INSPECT_STEPS = [
    ("start", "Starting Python and SpikeInterface"),
    ("loading", "Loading the recording"),
    ("predictions", "Getting the model predictions"),
    ("views", "Preparing the SpikeInterface-GUI views"),
]

# after this many seconds, explain why it can take long
SLOW_AFTER_SECONDS = 30


class OpeningDialog(QtWidgets.QDialog):
    """
    Shown while the SpikeInterface-GUI opens: the steps it goes through, the time so far, how long it
    took last time, and a Cancel button. Closed by the main window once the SpikeInterface-GUI is shown.
    """

    cancelled = pyqtSignal()

    def __init__(self, title, steps, previous_seconds=None, parent=None, window_title="Opening the SpikeInterface-GUI",
                 note=None, slow_message=None):
        """`note` (text) replaces the help about the SpikeInterface-GUI; `slow_message` is shown when it takes long."""

        super().__init__(parent)

        self.steps = steps
        self.previous_seconds = previous_seconds
        self.start_time = time.time()
        self.step_start_time = self.start_time
        self.current_step = 0

        self.slow_message = slow_message
        self.setWindowTitle(window_title)
        # no close button: closing would not stop the SpikeInterface-GUI from opening, Cancel does
        self.setWindowFlags(Qt.Dialog | Qt.CustomizeWindowHint | Qt.WindowTitleHint)
        self.setMinimumWidth(540)
        self.setStyleSheet("background-color: white;")

        layout = QtWidgets.QVBoxLayout(self)
        layout.setContentsMargins(24, 20, 24, 20)
        layout.setSpacing(10)

        title_label = QtWidgets.QLabel(title)
        title_label.setStyleSheet("font-weight: bold; font-size: 13pt;")
        title_label.setWordWrap(True)
        layout.addWidget(title_label)

        self.step_labels = []
        for _, step_text in steps:
            step_label = QtWidgets.QLabel()
            layout.addWidget(step_label)
            self.step_labels.append(step_label)

        progress_layout = QtWidgets.QHBoxLayout()
        self.progress_bar = QtWidgets.QProgressBar()
        self.progress_bar.setTextVisible(False)
        self.progress_bar.setFixedHeight(10)
        progress_layout.addWidget(self.progress_bar, 1)
        self.time_label = QtWidgets.QLabel()
        progress_layout.addWidget(self.time_label)
        layout.addLayout(progress_layout)

        self.message_label = QtWidgets.QLabel()
        self.message_label.setWordWrap(True)
        self.message_label.setStyleSheet("color: '#555555';")
        layout.addWidget(self.message_label)

        if note is None:
            # what to do once it is open, read while waiting
            help_box = sigui_help_box()
            # wrapped text gets too little height in a dialog, so reserve the height it needs at the dialog's width
            help_box.ensurePolished()
            help_box.setMinimumHeight(help_box.heightForWidth(self.minimumWidth() - 48) + 24)
            layout.addWidget(help_box)
        else:
            note_label = QtWidgets.QLabel(note)
            note_label.setWordWrap(True)
            note_label.setTextFormat(Qt.TextFormat.RichText)
            layout.addWidget(note_label)

        cancel_layout = QtWidgets.QHBoxLayout()
        cancel_layout.addStretch()
        cancel_button = QtWidgets.QPushButton("Cancel")
        cancel_button.clicked.connect(self.cancelled.emit)
        cancel_layout.addWidget(cancel_button)
        layout.addLayout(cancel_layout)

        self.timer = QTimer(self)
        self.timer.timeout.connect(self.update_display)
        self.timer.start(200)
        self.update_display()

    def set_step(self, step_name):
        step_names = [name for name, _ in self.steps]
        if step_name in step_names:
            self.current_step = step_names.index(step_name)
            self.step_start_time = time.time()
            self.update_display()

    def update_display(self):

        for index, ((_, step_text), step_label) in enumerate(zip(self.steps, self.step_labels)):
            if index < self.current_step:
                step_label.setText(f"<span style='color:#2E7D32'>✓</span>&nbsp;&nbsp;{step_text}")
            elif index == self.current_step:
                step_label.setText(f"<span style='color:#1565C0'>●</span>&nbsp;&nbsp;<b>{step_text}...</b>")
            else:
                step_label.setText(f"<span style='color:#AAAAAA'>○&nbsp;&nbsp;{step_text}</span>")

        elapsed = time.time() - self.start_time

        # the bar jumps with each step, and creeps forward within a step (slower and slower, never
        # reaching the next step) so it doesn't look stuck while a step takes long
        time_in_step = time.time() - self.step_start_time
        step_fraction = 0.9 * (1 - math.exp(-time_in_step / 8))
        self.progress_bar.setMaximum(1000)
        self.progress_bar.setValue(int(1000 * (self.current_step + step_fraction) / len(self.steps)))

        time_text = f"{elapsed:.0f} s"
        if self.previous_seconds is not None:
            time_text += f"  (last time: about {self.previous_seconds:.0f} s)"
        self.time_label.setText(time_text)

        if self.slow_message is not None:
            self.message_label.setText(self.slow_message)
        elif elapsed > SLOW_AFTER_SECONDS and (self.previous_seconds is None or elapsed > self.previous_seconds * 1.5):
            self.message_label.setText(
                "Still working. Large recordings take longer to load, and the first time a recording is opened "
                "some data for the views may be computed. Nothing has crashed, please don't close anything."
            )
        else:
            self.message_label.setText(
                "UnitRefine is still working, nothing has crashed. Please don't close anything; "
                "large recordings can take a minute."
            )

    def done_opening(self):
        self.timer.stop()
        self.accept()

    def elapsed_seconds(self):
        return time.time() - self.start_time
