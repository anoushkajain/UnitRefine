# The SpikeInterface-GUI scripts run in their own process. They print these markers so the main
# window can show which step they are on while they open, and when their window is shown.
SIGUI_SHOWN_MARKER = "UNITREFINE_SIGUI_WINDOW_SHOWN"
SIGUI_STEP_MARKER = "UNITREFINE_SIGUI_STEP:"
# printed when labels could not be saved where they belong and were saved to a backup file instead
LABELS_BACKUP_MARKER = "UNITREFINE_LABELS_BACKUP:"


# The labels are passed to the SpikeInterface-GUI in curation format 2, which spikeinterface-gui handles
# correctly (also for units shown without a label) from this version on
MINIMUM_SIGUI_VERSION = "0.14.0"


def check_sigui_version():
    """
    Stops the SpikeInterface-GUI script with a clear message if spikeinterface-gui is too old: older versions
    would not save the labels set with the n/g/m keys correctly.
    """
    import sys
    from packaging.version import parse
    import spikeinterface_gui

    installed = getattr(spikeinterface_gui, "__version__", "0")
    if parse(installed) < parse(MINIMUM_SIGUI_VERSION):
        print(f"UnitRefine needs spikeinterface-gui {MINIMUM_SIGUI_VERSION} or newer, but {installed} is installed, "
              f"which would not save your labels correctly. Please upgrade it, e.g. with "
              f"'pip install --upgrade spikeinterface-gui' or 'uv sync'.", flush=True)
        sys.exit(1)


def unwritable_file_message(path, error):
    """What to tell the user when a file can't be written, e.g. because it is read-only."""
    advice = ("The file is read-only or open in another program. In File Explorer, right-click it, choose "
              "Properties and untick 'Read-only' (or close the other program), then try again.")
    return f"UnitRefine could not save {path}:\n{error}\n\n{advice}"


def save_labels_or_backup(labels, path, write=None, **to_csv_kwargs):
    """
    Saves the labels set in the SpikeInterface-GUI to `path` (with `write(labels, path)` if given). If that
    fails, e.g. because the file is read-only, the labels are saved to a backup file in the home folder, so
    they are not lost, and the main window is told where (LABELS_BACKUP_MARKER). Returns where they were saved.
    """
    import time
    from pathlib import Path

    path = Path(path)
    try:
        if write is not None:
            write(labels, path)
        else:
            labels.to_csv(path, **to_csv_kwargs)
        return path
    except OSError as error:
        backup = Path.home() / f"UnitRefine_unsaved_{time.strftime('%Y%m%d_%H%M%S')}_{path.name}"
        labels.to_csv(backup, **to_csv_kwargs)
        print(unwritable_file_message(path, error), flush=True)
        print(f"Your labels were saved to {backup} instead. Once {path.name} can be written, copy that file over it.",
              flush=True)
        print(f"{LABELS_BACKUP_MARKER}{backup}|{path}", flush=True)
        return backup


def report_sigui_step(step_name):
    """Called by the SpikeInterface-GUI scripts when they start a step, e.g. 'loading'."""
    print(f"{SIGUI_STEP_MARKER}{step_name}", flush=True)
