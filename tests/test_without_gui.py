"""
Applying models when UnitRefine is installed without the GUI (`uv sync --no-group gui`, or `pip install unitrefine`),
e.g. on a computing cluster: nothing here may need Qt, pyqtgraph or spikeinterface-gui. The GitHub workflow runs
this file in such an install (the without-gui job); in a full install it runs with the other tests.
"""

import importlib.util
import subprocess
import sys

import pandas as pd
import pytest

from conftest import make_analyzer, write_labels


@pytest.fixture(scope="module")
def headless_model(tmp_path_factory):
    """A SUA vs all model trained with SpikeInterface (the Train window needs the GUI), and a new recording."""
    import spikeinterface.full as si
    from spikeinterface.curation import train_model

    folder = tmp_path_factory.mktemp("headless")
    training = make_analyzer(folder / "training_analyzer", num_units=30, seed=0)
    analyzer = si.load_sorting_analyzer(training, load_extensions=False)
    labels = write_labels(folder / "labels.csv", analyzer.unit_ids)
    train_model(
        mode="analyzers", analyzers=[analyzer], folder=folder / "models" / "sua_vs_all",
        labels=[["good" if labels[str(unit_id)] == "good" else "not-sua" for unit_id in analyzer.unit_ids]],
        imputation_strategies=["median"], scaling_techniques=["standard_scaler"], classifiers=["RandomForestClassifier"],
        seed=0,
    )
    return dict(model=folder / "models" / "sua_vs_all", new=make_analyzer(folder / "new_analyzer", num_units=25, seed=1),
                folder=folder)


def test_models_are_applied_without_the_gui(headless_model, tmp_path):
    """In a separate Python, so the check that no GUI module is loaded isn't affected by the other tests."""
    output = tmp_path / "predictions"
    script = (
        "import sys\n"
        "from unitrefine.cli import apply_main\n"
        f"status = apply_main([{str(headless_model['folder'])!r}, '--model', {str(headless_model['model'])!r}, "
        f"'--output', {str(output)!r}, '--compute_missing_metrics'])\n"
        "gui = [name for name in sys.modules if name.split('.')[0] in ('PyQt5', 'pyqtgraph', 'spikeinterface_gui', 'shap', 'matplotlib')]\n"
        "print('GUI_MODULES', gui)\n"
        "sys.exit(status if len(gui) == 0 else 3)\n"
    )
    result = subprocess.run([sys.executable, "-c", script], capture_output=True, text=True)
    assert result.returncode == 0, result.stdout[-3000:] + result.stderr[-3000:]
    # both analyzers in the folder are found and labelled
    for name in ["training_analyzer", "new_analyzer"]:
        predictions = pd.read_csv(output / f"{name}__labels_from_sua_vs_all.csv")
        assert set(predictions["sua_vs_all_model_label"]) <= {"good", "not-sua"}


@pytest.mark.skipif(importlib.util.find_spec("PyQt5") is not None, reason="the GUI is installed")
def test_the_gui_says_how_to_install_it():
    result = subprocess.run([sys.executable, "-m", "unitrefine.launcher", "--project_folder", "unused"],
                            capture_output=True, text=True)
    assert result.returncode != 0
    assert 'pip install "unitrefine[gui]"' in result.stderr
