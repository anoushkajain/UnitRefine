"""
Starts UnitRefine. An "Opening UnitRefine" box appears at once, while the slow parts run behind it: loading
SpikeInterface and the machine learning libraries, loading the project, and building the window with its
saved figures. The window is only shown once it is complete, so it doesn't flicker while it fills in.

Only PyQt5 is imported at the top of this file, so the box can appear before the slow imports.
"""

import sys
import time
from argparse import ArgumentParser
from pathlib import Path

import PyQt5.QtWidgets as QtWidgets
from PyQt5.QtCore import QCoreApplication, QEvent, Qt
from PyQt5.QtGui import QFontDatabase, QIcon, QPixmap
from PyQt5.QtWidgets import QStyleFactory

from unitrefine import theme

RESOURCES_FOLDER = Path(__file__).absolute().parent / "resources"

STARTUP_STEPS = [
    "Loading SpikeInterface and the machine learning libraries",
    "Loading the project",
    "Building the window and loading the saved figures",
]


def enable_high_dpi_scaling():
    """
    Makes the GUI follow the display scaling set in the operating system (e.g. 150% on a laptop with a
    high resolution screen), so it has the same physical size on every screen. Must be called before
    the QApplication is created.
    """
    QtWidgets.QApplication.setAttribute(Qt.AA_EnableHighDpiScaling, True)
    QtWidgets.QApplication.setAttribute(Qt.AA_UseHighDpiPixmaps, True)
    # use the exact scaling (e.g. 1.25) instead of rounding it to a whole number
    QtWidgets.QApplication.setHighDpiScaleFactorRoundingPolicy(Qt.HighDpiScaleFactorRoundingPolicy.PassThrough)


def set_app_font(app):
    """
    Uses Inter, which is included in resources/fonts so the GUI looks the same on every system.
    If it can't be loaded, falls back to Segoe UI (Windows) or the system's default font.
    """
    loaded_families = set()
    for font_file in (RESOURCES_FOLDER / "fonts").glob("*.ttf"):
        font_id = QFontDatabase.addApplicationFont(str(font_file))
        if font_id != -1:
            loaded_families.update(QFontDatabase.applicationFontFamilies(font_id))

    custom_font = app.font()
    if "Inter" in loaded_families:
        custom_font.setFamily("Inter")
    elif "Segoe UI" in QFontDatabase().families():
        custom_font.setFamily("Segoe UI")
    custom_font.setPointSize(10)
    app.setFont(custom_font)


class StartupDialog(QtWidgets.QWidget):
    """The 'Opening UnitRefine' box: the steps of opening, which is done, and the time so far."""

    def __init__(self, project_folder):
        super().__init__(None, Qt.SplashScreen | Qt.FramelessWindowHint | Qt.WindowStaysOnTopHint)
        self.setAttribute(Qt.WidgetAttribute.WA_TranslucentBackground)
        self.start_time = time.time()
        self.current_step = -1

        outer_layout = QtWidgets.QVBoxLayout(self)
        outer_layout.setContentsMargins(0, 0, 0, 0)
        card = QtWidgets.QFrame()
        card.setObjectName("startup_card")
        card.setStyleSheet(
            f"QFrame#startup_card {{ background-color: white; border: 1px solid {theme.BORDER}; border-radius: 14px; }}"
        )
        outer_layout.addWidget(card)
        layout = QtWidgets.QVBoxLayout(card)
        layout.setContentsMargins(28, 24, 28, 22)
        layout.setSpacing(10)

        header_layout = QtWidgets.QHBoxLayout()
        logo_file = RESOURCES_FOLDER / "logo.png"
        if logo_file.exists():
            logo = QtWidgets.QLabel()
            logo.setPixmap(QPixmap(str(logo_file)).scaled(44, 44, Qt.KeepAspectRatio, Qt.SmoothTransformation))
            header_layout.addWidget(logo)
        title_layout = QtWidgets.QVBoxLayout()
        title_layout.setSpacing(0)
        title = QtWidgets.QLabel("Opening UnitRefine")
        title.setStyleSheet("font-weight: bold; font-size: 15pt;")
        title_layout.addWidget(title)
        project_label = QtWidgets.QLabel(f"Project: {Path(project_folder).name}")
        project_label.setObjectName("muted")
        title_layout.addWidget(project_label)
        header_layout.addLayout(title_layout, 1)
        layout.addLayout(header_layout)

        self.step_labels = []
        for step_text in STARTUP_STEPS:
            step_label = QtWidgets.QLabel()
            step_label.setTextFormat(Qt.TextFormat.RichText)
            layout.addWidget(step_label)
            self.step_labels.append(step_label)

        self.progress_bar = QtWidgets.QProgressBar()
        self.progress_bar.setRange(0, len(STARTUP_STEPS))
        self.progress_bar.setTextVisible(False)
        self.progress_bar.setFixedHeight(8)
        layout.addWidget(self.progress_bar)

        self.time_label = QtWidgets.QLabel()
        self.time_label.setObjectName("muted")
        layout.addWidget(self.time_label)

        self.setFixedWidth(500)
        self.update_display()

    def set_step(self, step_index):
        self.current_step = step_index
        self.update_display()

    def update_display(self):
        for index, (step_text, step_label) in enumerate(zip(STARTUP_STEPS, self.step_labels)):
            if index < self.current_step:
                step_label.setText(f"<span style='color:{theme.SUCCESS}'>✓</span>&nbsp; {step_text}")
                step_label.setStyleSheet(f"color: {theme.MUTED};")
            elif index == self.current_step:
                step_label.setText(f"<span style='color:{theme.ACCENT}'>●</span>&nbsp; <b>{step_text}…</b>")
                step_label.setStyleSheet("")
            else:
                step_label.setText(f"○&nbsp; {step_text}")
                step_label.setStyleSheet("color: #98A2B3;")
        self.progress_bar.setValue(max(self.current_step, 0))
        self.time_label.setText(f"{time.time() - self.start_time:.0f} s · the first start after installing can take a minute")
        # draw now: the slow steps run in this thread, so the box can't redraw itself while they run
        for _ in range(3):
            QtWidgets.QApplication.processEvents()


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

    enable_high_dpi_scaling()
    app = QtWidgets.QApplication(sys.argv)
    app.setStyle(QStyleFactory.create("Fusion"))
    app.setStyleSheet(theme.STYLE_SHEET)
    icon_file = RESOURCES_FOLDER / "logo.png"
    if icon_file.exists():
        app.setWindowIcon(QIcon(str(icon_file)))
    set_app_font(app)

    startup = StartupDialog(project_folder)
    startup.show()
    startup.set_step(0)

    from unitrefine.main import APP_DESCRIPTION, MainWindow, UnitRefineProject, initial_window_size, load_project

    startup.set_step(1)
    if project_folder.is_dir():
        print("Project already exists. Loading...")
        project = load_project(project_folder)
    else:
        print("Project Folder does not exist. Creating now...")
        project = UnitRefineProject(project_folder)
        project.save()

    startup.set_step(2)
    window = MainWindow(project)
    window.setWindowTitle(f"UnitRefine: {APP_DESCRIPTION}")
    window.resize(initial_window_size(app.primaryScreen().availableGeometry()))
    window.finish_loading()

    startup.set_step(len(STARTUP_STEPS))
    window.show()
    QtWidgets.QApplication.processEvents()
    startup.close()
    app.exec()


if __name__ == "__main__":
    main()
