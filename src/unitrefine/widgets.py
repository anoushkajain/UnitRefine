"""
Widgets of the main window: collapsible section cards, rows with a shortened path, and the figure
galleries, whose figures are made one at a time in a background thread so the window never freezes.
"""

import itertools
import queue
import shutil
import threading
import time
from pathlib import Path

import PyQt5.QtWidgets as QtWidgets
from PyQt5.QtCore import QObject, QPoint, QRect, QSize, Qt, QTimer, QUrl, pyqtSignal
from PyQt5.QtGui import QDesktopServices, QPainter, QPixmap

from unitrefine import figures, theme

THUMBNAIL_SIZE = QSize(232, 150)


def open_folder(folder):
    folder = Path(folder)
    folder.mkdir(parents=True, exist_ok=True)
    QDesktopServices.openUrl(QUrl.fromLocalFile(str(folder)))


def muted_label(text, word_wrap=True):
    label = QtWidgets.QLabel(text)
    label.setObjectName("muted")
    label.setWordWrap(word_wrap)
    label.setTextFormat(Qt.TextFormat.RichText)
    return label


def subheading(text):
    label = QtWidgets.QLabel(text)
    label.setObjectName("subheading")
    return label


def chip(text, done=False):
    """A small rounded label for a status, e.g. 795/796 labelled."""
    label = QtWidgets.QLabel(text)
    label.setObjectName("chip_done" if done else "chip")
    label.setAlignment(Qt.AlignmentFlag.AlignCenter)
    return label


def clear_layout(layout):
    """Removes and deletes everything in a layout."""
    while layout.count() > 0:
        item = layout.takeAt(0)
        widget = item.widget()
        if widget is not None:
            # hide it at once: deleteLater only deletes it once Qt gets back to its event loop
            widget.hide()
            widget.deleteLater()
        elif item.layout() is not None:
            clear_layout(item.layout())


SIGUI_HELP = (
    "<b>Labelling in the SpikeInterface-GUI</b><br>"
    "• Select a unit and press <b>n</b> (noise), <b>g</b> (good) or <b>m</b> (MUA). The label shows in the "
    "<i>quality</i> column.<br>"
    "• To change a label, press another key: it replaces the old one.<br>"
    "• <b>Your labels are saved automatically when you close the SpikeInterface-GUI.</b>"
)


def sigui_help_box():
    """How to label in the SpikeInterface-GUI, and that labels are saved when it is closed."""
    box = QtWidgets.QLabel(SIGUI_HELP)
    box.setWordWrap(True)
    box.setTextFormat(Qt.TextFormat.RichText)
    box.setStyleSheet(
        f"background-color: {theme.SIGUI_BUTTON['background']}; border: 1px solid {theme.SIGUI_BUTTON['border']}; "
        f"border-radius: 8px; padding: 8px 12px; color: #2D2A6E;"
    )
    return box


def list_box():
    """A white box for a list of rows, like a table on the section's tint. Returns the box and its layout."""
    box = QtWidgets.QFrame()
    box.setObjectName("list")
    rows_layout = QtWidgets.QVBoxLayout(box)
    rows_layout.setContentsMargins(0, 2, 0, 2)
    rows_layout.setSpacing(0)
    return box, rows_layout


def list_message(text):
    """A message in a list of rows, e.g. when it is empty."""
    label = muted_label(text)
    label.setContentsMargins(12, 10, 12, 10)
    return label


def group_label(text):
    """The title of a group of rows, e.g. 'Analyzers used during training the models:'."""
    label = QtWidgets.QLabel(text)
    label.setStyleSheet("color: #344054; padding: 10px 12px 4px 12px; font-weight: bold; font-size: 9pt;")
    return label


class BusyNotice:
    """
    Shows what a slow action is doing while it runs, e.g. `with BusyNotice(window, "Retraining", "..."):`.
    The window can't update while the action runs, so this tells the user it is working and not stuck.
    """

    def __init__(self, parent, title, message):
        self.parent = parent
        self.title = title
        self.message = message

    def __enter__(self):
        self.dialog = QtWidgets.QDialog(self.parent, Qt.Dialog | Qt.CustomizeWindowHint | Qt.WindowTitleHint)
        self.dialog.setWindowTitle("UnitRefine is working")
        self.dialog.setModal(True)
        self.dialog.setMinimumWidth(440)
        layout = QtWidgets.QVBoxLayout(self.dialog)
        layout.setContentsMargins(24, 20, 24, 20)
        layout.setSpacing(10)
        title = QtWidgets.QLabel(self.title)
        title.setStyleSheet("font-weight: bold; font-size: 12pt;")
        title.setWordWrap(True)
        layout.addWidget(title)
        self.message_label = muted_label(self.message)
        layout.addWidget(self.message_label)
        bar = QtWidgets.QProgressBar()
        bar.setRange(0, 0)
        bar.setFixedHeight(6)
        bar.setTextVisible(False)
        layout.addWidget(bar)
        self.dialog.show()
        QtWidgets.QApplication.setOverrideCursor(Qt.CursorShape.WaitCursor)
        self.process_events()
        return self

    def update(self, message):
        self.message_label.setText(message)
        self.process_events()

    def process_events(self):
        # draw the notice before the work starts; it is modal, so no other button can be clicked meanwhile
        for _ in range(3):
            QtWidgets.QApplication.processEvents()

    def __exit__(self, *exception):
        QtWidgets.QApplication.restoreOverrideCursor()
        self.dialog.close()
        self.dialog.deleteLater()
        return False


class ElidedLabel(QtWidgets.QLabel):
    """Shows as much of a long text (e.g. a path) as fits, cut at the start; the full text is in the tooltip."""

    def __init__(self, text="", parent=None, elide_mode=Qt.TextElideMode.ElideLeft):
        super().__init__(parent)
        self.full_text = ""
        self.elide_mode = elide_mode
        self.setSizePolicy(QtWidgets.QSizePolicy.Expanding, QtWidgets.QSizePolicy.Preferred)
        self.set_full_text(text)

    def set_full_text(self, text):
        self.full_text = str(text)
        self.setToolTip(self.full_text)
        self.update_text()

    def minimumSizeHint(self):
        return QSize(60, super().minimumSizeHint().height())

    def sizeHint(self):
        return QSize(self.fontMetrics().horizontalAdvance(self.full_text) + 4, super().sizeHint().height())

    def resizeEvent(self, event):
        super().resizeEvent(event)
        self.update_text()

    def update_text(self):
        self.setText(self.fontMetrics().elidedText(self.full_text, self.elide_mode, max(self.width(), 20)))


class ClickableWidget(QtWidgets.QWidget):
    clicked = pyqtSignal()

    def mouseReleaseEvent(self, event):
        if event.button() == Qt.MouseButton.LeftButton and self.rect().contains(event.pos()):
            self.clicked.emit()
        super().mouseReleaseEvent(event)


class CollapsibleSection(QtWidgets.QFrame):
    """A white card with a numbered header which folds the section's content away when clicked."""

    toggled = pyqtSignal(bool)

    def __init__(self, number, title, summary="", expanded=True, parent=None):
        super().__init__(parent)
        section_theme = theme.SECTION_THEMES.get(number, theme.SECTION_THEMES[1])
        self.setObjectName("section")
        self.setStyleSheet(
            f"QFrame#section {{ background-color: {section_theme['tint']}; border: 1px solid {section_theme['border']}; "
            "border-radius: 12px; }"
        )

        layout = QtWidgets.QVBoxLayout(self)
        layout.setContentsMargins(20, 14, 20, 16)
        layout.setSpacing(0)

        header = ClickableWidget()
        header.setCursor(Qt.CursorShape.PointingHandCursor)
        header.setToolTip("Click to fold or unfold this section")
        header_layout = QtWidgets.QHBoxLayout(header)
        header_layout.setContentsMargins(0, 0, 0, 0)
        header_layout.setSpacing(12)

        badge = QtWidgets.QLabel(str(number))
        badge.setFixedSize(30, 30)
        badge.setAlignment(Qt.AlignmentFlag.AlignCenter)
        badge.setStyleSheet(
            f"background-color: {section_theme['accent']}; color: white; "
            "border-radius: 15px; font-weight: bold; font-size: 11pt;"
        )
        header_layout.addWidget(badge)

        title_label = QtWidgets.QLabel(title)
        title_label.setStyleSheet("font-weight: bold; font-size: 14pt;")
        header_layout.addWidget(title_label)

        self.summary_label = ElidedLabel(summary, elide_mode=Qt.TextElideMode.ElideRight)
        self.summary_label.setObjectName("muted")
        self.summary_label.setStyleSheet("padding-top: 3px;")
        header_layout.addWidget(self.summary_label, 1)

        self.chevron = QtWidgets.QLabel()
        self.chevron.setObjectName("muted")
        self.chevron.setStyleSheet("font-size: 13pt;")
        header_layout.addWidget(self.chevron)
        header.clicked.connect(lambda: self.set_expanded(not self.expanded))
        layout.addWidget(header)

        self.body = QtWidgets.QWidget()
        self.body_layout = QtWidgets.QVBoxLayout(self.body)
        self.body_layout.setContentsMargins(0, 14, 0, 0)
        self.body_layout.setSpacing(10)
        layout.addWidget(self.body)

        self.expanded = True
        self.set_expanded(expanded, emit=False)

    def set_expanded(self, expanded, emit=True):
        self.expanded = expanded
        self.body.setVisible(expanded)
        self.chevron.setText("▾" if expanded else "▸")
        if emit:
            self.toggled.emit(expanded)

    def set_summary(self, text):
        self.summary_label.set_full_text(text)


# analyzer folders with these names don't say which recording they are, so their parent folder is shown too
GENERIC_FOLDER_NAMES = {"analyzer", "sorting_analyzer", "sortinganalyzer", "sorting", "si_analyzer", "output", "results", "an"}


def recording_name_and_location(path):
    """
    A short name for a recording's analyzer, e.g. 'mouse1_session2', with the folder it is in. Generic folder
    names (e.g. sorting_analyzer) get their parent folder's name added: 'session2 / sorting_analyzer'.
    """
    text = str(path).rstrip("/\\")
    parts = [part for part in text.replace("\\", "/").split("/") if part != ""]
    if len(parts) == 0:
        return text, ""
    name = parts[-1]
    location_parts = parts[:-1]
    if (name.lower() in GENERIC_FOLDER_NAMES or len(name) <= 3) and len(parts) > 1:
        name = f"{parts[-2]} / {name}"
        location_parts = parts[:-2]
    location = "/".join(location_parts)
    if "://" in text:
        location = text[: text.index("://") + 3] + "/".join(location_parts[1:])
    return name, location


class PathRow(QtWidgets.QFrame):
    """
    A row of a list: a recording's short name in bold with the folder it is in below it (the full path is
    in the tooltip), then widgets on the right such as buttons and counts.
    """

    def __init__(self, path, widgets, first=False, parent=None):
        super().__init__(parent)
        self.setObjectName("row")
        self.setStyleSheet("QFrame#row { background: transparent; border: none; "
                           + ("" if first else f"border-top: 1px solid {theme.BORDER};") + " }")
        layout = QtWidgets.QHBoxLayout(self)
        layout.setContentsMargins(12, 7, 10, 7)
        layout.setSpacing(8)

        name, location = recording_name_and_location(path)
        text_layout = QtWidgets.QVBoxLayout()
        text_layout.setSpacing(0)
        self.name_label = ElidedLabel(name, elide_mode=Qt.TextElideMode.ElideMiddle)
        self.name_label.setObjectName("row_name")
        self.name_label.setToolTip(str(path))
        text_layout.addWidget(self.name_label)
        self.location_label = ElidedLabel(location)
        self.location_label.setObjectName("row_location")
        self.location_label.setToolTip(str(path))
        text_layout.addWidget(self.location_label)
        self.location_label.setVisible(location != "")
        layout.addLayout(text_layout, 1)
        for widget in widgets:
            layout.addWidget(widget)


class FlowLayout(QtWidgets.QLayout):
    """Places widgets left to right and wraps them onto new lines, like words in a paragraph."""

    def __init__(self, parent=None, spacing=12):
        super().__init__(parent)
        self.items = []
        self.setSpacing(spacing)
        self.setContentsMargins(0, 0, 0, 0)

    def addItem(self, item):
        self.items.append(item)

    def count(self):
        return len(self.items)

    def itemAt(self, index):
        return self.items[index] if 0 <= index < len(self.items) else None

    def takeAt(self, index):
        return self.items.pop(index) if 0 <= index < len(self.items) else None

    def expandingDirections(self):
        return Qt.Orientations(0)

    def hasHeightForWidth(self):
        return True

    def heightForWidth(self, width):
        return self.do_layout(QRect(0, 0, width, 0), apply=False)

    def setGeometry(self, rect):
        super().setGeometry(rect)
        self.do_layout(rect, apply=True)

    def sizeHint(self):
        return self.minimumSize()

    def minimumSize(self):
        size = QSize()
        for item in self.items:
            size = size.expandedTo(item.minimumSize())
        return size

    def do_layout(self, rect, apply):
        x, y, line_height = rect.x(), rect.y(), 0
        for item in self.items:
            item_size = item.sizeHint()
            if x + item_size.width() > rect.right() + 1 and line_height > 0:
                x = rect.x()
                y += line_height + self.spacing()
                line_height = 0
            if apply:
                item.setGeometry(QRect(QPoint(x, y), item_size))
            x += item_size.width() + self.spacing()
            line_height = max(line_height, item_size.height())
        return y + line_height - rect.y()


# --------------------------------------------------------------------------------------------------
# making figures in the background
# --------------------------------------------------------------------------------------------------

class FigureJob:
    _ids = itertools.count()

    def __init__(self, spec, priority):
        self.id = next(self._ids)
        self.spec = spec
        self.priority = priority


class FigureWorker(QObject):
    """
    Makes figures one at a time in a background thread. A figure asked for again before it was made only
    runs once (the latest request). Figures someone is waiting for (e.g. clicked on) go first.
    """

    job_started = pyqtSignal(object)
    job_progress = pyqtSignal(object, str)
    job_finished = pyqtSignal(object, object)

    def __init__(self, parent=None):
        super().__init__(parent)
        self.jobs = queue.PriorityQueue()
        self.latest_job_ids = {}
        self.order = itertools.count()
        threading.Thread(target=self.run, daemon=True, name="unitrefine-figures").start()

    def submit(self, spec, priority=1):
        job = FigureJob(spec, priority)
        self.latest_job_ids[str(spec.path)] = job.id
        self.jobs.put((priority, next(self.order), job))
        return job

    def run(self):
        try:
            while True:
                _, _, job = self.jobs.get()
                if self.latest_job_ids.get(str(job.spec.path)) != job.id:
                    continue
                self.job_started.emit(job)
                result = figures.render(job.spec, progress=self.progress_reporter(job))
                self.job_finished.emit(job, result)
        except RuntimeError:
            # the window was closed while a figure was being made: nothing is waiting for figures any more
            return

    def progress_reporter(self, job):
        def report(message):
            try:
                self.job_progress.emit(job, message)
            except RuntimeError:
                # the window was closed: let the figure finish without reporting
                pass
        return report


class FigureCard(QtWidgets.QFrame):
    """
    A thumbnail of one figure. While the figure is (re)made, the card keeps showing what it showed before
    (or is hidden, if it showed nothing yet); the new result is held back until the gallery shows all its
    new figures together, so the figures don't change one by one in front of the user.
    """

    clicked = pyqtSignal()
    changed = pyqtSignal()

    def __init__(self, spec, parent=None):
        super().__init__(parent)
        self.spec = spec
        # waiting / running while a job makes the figure, ready once it is made but not shown yet,
        # then the shown result's status: done, nodata or error
        self.state = "waiting"
        self.result = None          # the result shown
        self.pending_result = None  # a new result, shown when the gallery's other figures are ready too
        self.job_id = None
        self.pending_signature = None
        self.progress_message = ""
        self.start_time = None

        self.setObjectName("figure_card")
        self.setFixedSize(THUMBNAIL_SIZE.width() + 18, THUMBNAIL_SIZE.height() + 64)
        self.setCursor(Qt.CursorShape.PointingHandCursor)
        self.setStyleSheet(
            f"QFrame#figure_card {{ background-color: white; border: 1px solid {theme.BORDER}; border-radius: 10px; }}"
            f"QFrame#figure_card:hover {{ border: 1px solid {theme.ACCENT}; }}"
        )
        layout = QtWidgets.QVBoxLayout(self)
        layout.setContentsMargins(8, 8, 8, 8)
        layout.setSpacing(4)

        self.thumbnail = QtWidgets.QLabel()
        self.thumbnail.setFixedSize(THUMBNAIL_SIZE)
        self.thumbnail.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.thumbnail.setWordWrap(True)
        self.thumbnail.setStyleSheet("background-color: #F9FAFB; border-radius: 6px; color: #667085; padding: 8px;")
        layout.addWidget(self.thumbnail)

        self.title_label = QtWidgets.QLabel()
        self.title_label.setStyleSheet("font-weight: bold;")
        self.title_label.setText(self.title_label.fontMetrics().elidedText(spec.title, Qt.TextElideMode.ElideRight, THUMBNAIL_SIZE.width()))
        self.title_label.setToolTip(spec.title)
        layout.addWidget(self.title_label)

        self.status_label = QtWidgets.QLabel()
        self.status_label.setObjectName("muted")
        layout.addWidget(self.status_label)

    def mouseReleaseEvent(self, event):
        if event.button() == Qt.MouseButton.LeftButton:
            self.clicked.emit()
        super().mouseReleaseEvent(event)

    def is_busy(self):
        return self.state in ("waiting", "running")

    def has_something_to_show(self):
        return self.result is not None

    def set_waiting(self, job_id, pending_signature):
        self.job_id = job_id
        self.pending_signature = pending_signature
        self.state = "waiting"
        self.update_status()
        self.changed.emit()

    def set_running(self):
        self.state = "running"
        self.start_time = time.time()
        self.progress_message = "Starting"
        self.update_status()
        self.changed.emit()

    def set_progress(self, message):
        self.progress_message = message
        self.changed.emit()

    def set_finished(self, result):
        """The figure is made; it is shown by `show_pending_result`, once the gallery's other figures are made too."""
        self.job_id = None
        self.pending_signature = None
        self.pending_result = result
        self.state = "ready"
        self.changed.emit()

    def show_pending_result(self):
        if self.pending_result is not None:
            result, self.pending_result = self.pending_result, None
            self.set_result(result)

    def set_result(self, result):
        self.job_id = None
        self.pending_signature = None
        self.pending_result = None
        self.result = result
        self.state = result["status"]
        if self.state == "done":
            pixmap = QPixmap(str(self.spec.path))
            self.thumbnail.setText("")
            self.thumbnail.setPixmap(pixmap.scaled(
                THUMBNAIL_SIZE - QSize(8, 8), Qt.AspectRatioMode.KeepAspectRatio, Qt.TransformationMode.SmoothTransformation))
        elif self.state == "nodata":
            self.thumbnail.setPixmap(QPixmap())
            self.thumbnail.setText(result["message"])
        else:
            self.thumbnail.setPixmap(QPixmap())
            self.thumbnail.setText("Could not make this figure.\nClick for details.")
        self.update_status()
        self.changed.emit()

    def status_text(self):
        if self.state in ("waiting", "running", "ready"):
            if self.state == "running":
                return f"{self.progress_message}… {time.time() - self.start_time:.0f} s"
            return "Updating…"
        if self.state == "done":
            notes = self.result.get("notes", [])
            return f"{len(notes)} note{'s' if len(notes) > 1 else ''}, click to read" if len(notes) > 0 else "Click to enlarge"
        if self.state == "nodata":
            return "Not available yet"
        return "Error"

    def update_status(self):
        text = "Updating…" if self.state in ("waiting", "running", "ready") else self.status_text()
        self.status_label.setText(self.status_label.fontMetrics().elidedText(text, Qt.TextElideMode.ElideRight, THUMBNAIL_SIZE.width()))
        self.status_label.setToolTip(text)


class ZoomableImage(QtWidgets.QGraphicsView):
    """
    An image which fits the view at first, and can be zoomed with the mouse wheel (towards the point under the
    mouse) and moved by dragging. Double-click to fit it again.
    """

    zoom_changed = pyqtSignal()

    # zoom limits, relative to the image's own pixels
    MAX_ZOOM = 4.0

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setScene(QtWidgets.QGraphicsScene(self))
        self.item = QtWidgets.QGraphicsPixmapItem()
        self.item.setTransformationMode(Qt.TransformationMode.SmoothTransformation)
        self.scene().addItem(self.item)
        self.setDragMode(QtWidgets.QGraphicsView.DragMode.ScrollHandDrag)
        self.setTransformationAnchor(QtWidgets.QGraphicsView.ViewportAnchor.AnchorUnderMouse)
        self.setResizeAnchor(QtWidgets.QGraphicsView.ViewportAnchor.AnchorViewCenter)
        self.setRenderHints(QPainter.RenderHint.Antialiasing | QPainter.RenderHint.SmoothPixmapTransform)
        self.setFrameShape(QtWidgets.QFrame.NoFrame)
        self.setStyleSheet("background-color: white; border: none;")
        self.fitted = True

    def set_pixmap(self, pixmap):
        same_size = not self.item.pixmap().isNull() and self.item.pixmap().size() == pixmap.size()
        self.item.setPixmap(pixmap)
        self.scene().setSceneRect(self.item.boundingRect())
        # a new version of the same figure keeps the zoom; another figure starts fitted
        if self.fitted or not same_size:
            self.fit()

    def fit_scale(self):
        size = self.item.pixmap().size()
        if size.isEmpty():
            return 1.0
        viewport = self.viewport().size()
        return min(viewport.width() / size.width(), viewport.height() / size.height())

    def zoom(self):
        return self.transform().m11()

    def fit(self):
        if self.item.pixmap().isNull():
            return
        self.resetTransform()
        scale = self.fit_scale()
        self.scale(scale, scale)
        self.fitted = True
        self.zoom_changed.emit()

    def zoom_by(self, factor):
        if self.item.pixmap().isNull():
            return
        # not smaller than the whole figure, not larger than MAX_ZOOM times its pixels
        new_zoom = min(max(self.zoom() * factor, self.fit_scale()), max(self.MAX_ZOOM, self.fit_scale()))
        factor = new_zoom / self.zoom()
        self.scale(factor, factor)
        self.fitted = abs(new_zoom - self.fit_scale()) < 1e-6
        self.zoom_changed.emit()

    def wheelEvent(self, event):
        steps = event.angleDelta().y() / 120
        if steps != 0:
            self.zoom_by(1.25 ** steps)

    def mouseDoubleClickEvent(self, event):
        self.fit()

    def resizeEvent(self, event):
        super().resizeEvent(event)
        if self.fitted:
            self.fit()


class FigureViewer(QtWidgets.QDialog):
    """A large view of one figure, with how to read it and notes about its data. Follows the figure while it is made."""

    def __init__(self, card, parent=None):
        super().__init__(parent)
        self.card = card
        self.pixmap = None
        self.setWindowTitle(f"UnitRefine: {card.spec.title}")
        self.setAttribute(Qt.WidgetAttribute.WA_DeleteOnClose)

        layout = QtWidgets.QVBoxLayout(self)
        layout.setContentsMargins(20, 16, 20, 16)
        layout.setSpacing(10)

        title = QtWidgets.QLabel(card.spec.title)
        title.setStyleSheet("font-weight: bold; font-size: 14pt;")
        layout.addWidget(title)

        self.image_frame = QtWidgets.QFrame()
        self.image_frame.setObjectName("card")
        image_layout = QtWidgets.QVBoxLayout(self.image_frame)

        # zoom controls
        zoom_layout = QtWidgets.QHBoxLayout()
        zoom_layout.addWidget(muted_label("Scroll to zoom, drag to move, double-click to fit", word_wrap=False))
        zoom_layout.addStretch()
        self.zoom_label = muted_label("", word_wrap=False)
        zoom_layout.addWidget(self.zoom_label)
        self.zoom_buttons = []
        for text, tooltip, action in [
            ("−", "Zoom out", lambda: self.image_view.zoom_by(1 / 1.25)),
            ("+", "Zoom in", lambda: self.image_view.zoom_by(1.25)),
            ("Fit", "Show the whole figure", lambda: self.image_view.fit()),
        ]:
            button = theme.make_flat(QtWidgets.QPushButton(text))
            button.setToolTip(tooltip)
            button.clicked.connect(action)
            zoom_layout.addWidget(button)
            self.zoom_buttons.append(button)
        image_layout.addLayout(zoom_layout)

        # the figure, or a message while it is made or when it can't be made
        self.image_stack = QtWidgets.QStackedWidget()
        self.image_label = QtWidgets.QLabel()
        self.image_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.image_label.setWordWrap(True)
        self.image_label.setSizePolicy(QtWidgets.QSizePolicy.Ignored, QtWidgets.QSizePolicy.Ignored)
        self.image_stack.addWidget(self.image_label)
        self.image_view = ZoomableImage()
        self.image_view.zoom_changed.connect(self.update_zoom_label)
        self.image_stack.addWidget(self.image_view)
        self.image_stack.setMinimumSize(300, 200)
        image_layout.addWidget(self.image_stack, 1)
        self.busy_bar = QtWidgets.QProgressBar()
        self.busy_bar.setRange(0, 0)
        self.busy_bar.setFixedHeight(6)
        self.busy_bar.setTextVisible(False)
        image_layout.addWidget(self.busy_bar)
        layout.addWidget(self.image_frame, 1)

        self.caption_label = muted_label(card.spec.caption)
        layout.addWidget(self.caption_label)
        self.notes_label = QtWidgets.QLabel()
        self.notes_label.setWordWrap(True)
        self.notes_label.setTextFormat(Qt.TextFormat.RichText)
        self.notes_label.setTextInteractionFlags(Qt.TextInteractionFlag.TextSelectableByMouse)
        layout.addWidget(self.notes_label)

        buttons = QtWidgets.QHBoxLayout()
        self.saved_label = muted_label("", word_wrap=False)
        buttons.addWidget(self.saved_label, 1)
        folder_button = QtWidgets.QPushButton("Open figures folder")
        folder_button.clicked.connect(lambda: open_folder(card.spec.path.parent))
        self.save_button = QtWidgets.QPushButton("Save a copy…")
        self.save_button.clicked.connect(self.save_copy)
        close_button = theme.make_primary(QtWidgets.QPushButton("Close"))
        close_button.clicked.connect(self.close)
        for button in [folder_button, self.save_button, close_button]:
            buttons.addWidget(button)
        layout.addLayout(buttons)

        screen = QtWidgets.QApplication.primaryScreen().availableGeometry()
        self.resize(min(1250, int(screen.width() * 0.85)), min(950, int(screen.height() * 0.85)))

        card.changed.connect(self.update_from_card)
        self.update_from_card()

    def update_from_card(self):
        card = self.card
        busy = card.state in ("waiting", "running", "ready")
        self.busy_bar.setVisible(busy)
        shown = card.result
        self.save_button.setEnabled(shown is not None and shown["status"] == "done")
        self.pixmap = None
        if shown is None:
            self.image_label.setText(f"Making this figure: {card.status_text()}")
        elif shown["status"] == "done":
            self.pixmap = QPixmap(str(card.spec.path))
            self.saved_label.setText(f"Saved as {card.spec.path.name}" + ("  ·  a newer version is being made" if busy else ""))
        elif shown["status"] == "nodata":
            self.image_label.setText(shown["message"])
        else:
            self.image_label.setText(f"Could not make this figure:\n\n{shown['message']}\n\nThe terminal shows the details.")
        notes = shown.get("notes", []) if shown is not None else []
        self.notes_label.setText("".join(f"<p style='color:{theme.DANGER if note.startswith('Left out') else theme.TEXT}; margin: 2px 0;'>• {note}</p>" for note in notes))
        self.notes_label.setVisible(len(notes) > 0)

        has_image = self.pixmap is not None and not self.pixmap.isNull()
        if has_image:
            # only when the file changed, so a figure being remade doesn't reset the zoom
            modification_time = card.spec.path.stat().st_mtime_ns if card.spec.path.is_file() else None
            if modification_time != getattr(self, "shown_modification_time", None):
                self.image_view.set_pixmap(self.pixmap)
                self.shown_modification_time = modification_time
            self.image_stack.setCurrentWidget(self.image_view)
        else:
            self.image_stack.setCurrentWidget(self.image_label)
        for button in self.zoom_buttons:
            button.setEnabled(has_image)
        self.update_zoom_label()

    def update_zoom_label(self):
        if self.image_stack.currentWidget() is self.image_view:
            fit = self.image_view.fit_scale()
            self.zoom_label.setText(f"{self.image_view.zoom() / fit:.0%} of fitted size" if fit > 0 else "")
        else:
            self.zoom_label.setText("")

    def save_copy(self):
        path, _ = QtWidgets.QFileDialog.getSaveFileName(self, "Save figure", self.card.spec.path.name, "PNG image (*.png)")
        if path != "":
            shutil.copyfile(self.card.spec.path, path)

    def closeEvent(self, event):
        try:
            self.card.changed.disconnect(self.update_from_card)
        except TypeError:
            pass
        super().closeEvent(event)


class FigureGallery(QtWidgets.QWidget):
    """
    The figures of one section as thumbnails. Saved figures which are still up to date are shown at once.
    The others are made in the background, while a progress bar says what is being made; when they are all
    made, they are shown together. Figures shown already stay as they are until their new versions are shown.
    """

    def __init__(self, worker, folder, parent=None):
        super().__init__(parent)
        self.worker = worker
        self.folder = Path(folder)
        self.cards = {}
        self.specs = []

        layout = QtWidgets.QVBoxLayout(self)
        layout.setContentsMargins(0, 6, 0, 0)
        layout.setSpacing(8)

        header = QtWidgets.QHBoxLayout()
        header.addWidget(subheading("Figures"))
        self.status_label = muted_label("", word_wrap=False)
        header.addWidget(self.status_label, 1)
        refresh_button = theme.make_flat(QtWidgets.QPushButton("↻ Regenerate"))
        refresh_button.setToolTip("Make these figures again, even if nothing changed")
        refresh_button.clicked.connect(lambda: self.set_specs(self.specs, force=True))
        folder_button = theme.make_flat(QtWidgets.QPushButton("Open folder"))
        folder_button.setToolTip("Every figure is also saved as a png in this folder")
        folder_button.clicked.connect(lambda: open_folder(self.folder))
        header.addWidget(refresh_button)
        header.addWidget(folder_button)
        layout.addLayout(header)

        self.empty_label = muted_label("")
        layout.addWidget(self.empty_label)

        # shown while figures are made: what is being made, how many are done, and the time so far
        self.progress_frame = QtWidgets.QFrame()
        self.progress_frame.setObjectName("list")
        progress_layout = QtWidgets.QVBoxLayout(self.progress_frame)
        progress_layout.setContentsMargins(14, 10, 14, 12)
        progress_layout.setSpacing(6)
        self.progress_label = QtWidgets.QLabel()
        self.progress_label.setTextFormat(Qt.TextFormat.RichText)
        progress_layout.addWidget(self.progress_label)
        self.progress_bar = QtWidgets.QProgressBar()
        self.progress_bar.setTextVisible(False)
        self.progress_bar.setFixedHeight(8)
        progress_layout.addWidget(self.progress_bar)
        self.progress_frame.setVisible(False)
        layout.addWidget(self.progress_frame)
        self.batch_start_time = None
        self.progress_timer = QTimer(self)
        self.progress_timer.setInterval(500)
        self.progress_timer.timeout.connect(self.update_status)

        self.cards_widget = QtWidgets.QWidget()
        self.flow = FlowLayout(self.cards_widget)
        layout.addWidget(self.cards_widget)

        worker.job_started.connect(self.on_job_started)
        worker.job_progress.connect(self.on_job_progress)
        worker.job_finished.connect(self.on_job_finished)

    def set_specs(self, specs, folder=None, force=False, empty_message=""):
        """Shows these figures, reusing saved ones that are up to date (unless `force`) and making the others."""
        if folder is not None:
            self.folder = Path(folder)
        self.specs = list(specs)
        self.empty_label.setText(empty_message)
        self.empty_label.setVisible(len(self.specs) == 0 and empty_message != "")

        # keep the cards of figures which are still shown, so their thumbnails don't flicker
        old_cards = self.cards
        self.cards = {}
        while self.flow.count() > 0:
            self.flow.takeAt(0)
        for spec in self.specs:
            card = old_cards.pop(str(spec.path), None)
            if card is None:
                card = FigureCard(spec)
                card.clicked.connect(lambda card=card: self.open_viewer(card))
                card.changed.connect(self.update_status)
            card.spec = spec
            card.setParent(self.cards_widget)
            self.flow.addWidget(card)
            self.cards[str(spec.path)] = card
            self.update_card(card, force)
            # a figure only appears once it has something to show
            card.setVisible(card.has_something_to_show())
        for card in old_cards.values():
            card.hide()
            card.deleteLater()
        self.show_results_if_all_made()
        self.cards_widget.updateGeometry()
        self.update_status()

    def update_card(self, card, force=False):
        spec = card.spec
        if not force:
            cached = figures.cached_result(spec)
            if cached is not None:
                if card.result is None or card.result.get("signature") != cached["signature"] or card.state != cached["status"]:
                    card.set_result(cached)
                return
        current_signature = figures.signature(spec)
        # already being made from the same files: don't make it twice
        if not force and card.is_busy() and card.pending_signature == current_signature:
            return
        job = self.worker.submit(spec, priority=2 if spec.slow else 1)
        card.set_waiting(job.id, current_signature)

    def card_for_job(self, job):
        card = self.cards.get(str(job.spec.path))
        return card if card is not None and card.job_id == job.id else None

    def on_job_started(self, job):
        card = self.card_for_job(job)
        if card is not None:
            card.set_running()

    def on_job_progress(self, job, message):
        card = self.card_for_job(job)
        if card is not None:
            card.set_progress(message)

    def on_job_finished(self, job, result):
        card = self.card_for_job(job)
        if card is not None:
            card.set_finished(result)
            self.show_results_if_all_made()

    def show_results_if_all_made(self):
        """Once no figure of the gallery is being made, shows all the new ones together."""
        if any(card.is_busy() for card in self.cards.values()):
            return
        for card in self.cards.values():
            card.show_pending_result()
            card.setVisible(card.has_something_to_show())
        self.update_status()

    def open_viewer(self, card):
        if card.state == "waiting":
            # someone is waiting for it: make it next
            job = self.worker.submit(card.spec, priority=0)
            card.set_waiting(job.id, card.pending_signature)
        viewer = FigureViewer(card, parent=self.window())
        viewer.show()

    def update_status(self):
        cards = list(self.cards.values())
        in_batch = [card for card in cards if card.state in ("waiting", "running", "ready")]
        busy = [card for card in in_batch if card.is_busy()]
        if len(busy) > 0:
            if self.batch_start_time is None:
                self.batch_start_time = time.time()
                self.progress_timer.start()
            running = [card for card in busy if card.state == "running"]
            current = f"{running[0].spec.title}: {running[0].progress_message}…" if len(running) > 0 else "Waiting…"
            num_made = len(in_batch) - len(busy)
            self.progress_label.setText(
                f"<b>Making the figures: {num_made} of {len(in_batch)}</b> "
                f"<span style='color:{theme.MUTED}'>· {current} · {time.time() - self.batch_start_time:.0f} s · "
                "you can keep working</span>"
            )
            self.progress_bar.setRange(0, len(in_batch))
            self.progress_bar.setValue(num_made)
            self.progress_frame.setVisible(True)
            self.status_label.setText("")
        else:
            self.batch_start_time = None
            self.progress_timer.stop()
            self.progress_frame.setVisible(False)
            self.status_label.setText("Up to date · click a figure to enlarge it" if len(cards) > 0 else "")
