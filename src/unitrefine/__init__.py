# The SpikeInterface-GUI scripts run in their own process. They print these markers so the main
# window can show which step they are on while they open, and when their window is shown.
SIGUI_SHOWN_MARKER = "UNITREFINE_SIGUI_WINDOW_SHOWN"
SIGUI_STEP_MARKER = "UNITREFINE_SIGUI_STEP:"


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


def report_sigui_step(step_name):
    """Called by the SpikeInterface-GUI scripts when they start a step, e.g. 'loading'."""
    print(f"{SIGUI_STEP_MARKER}{step_name}", flush=True)
