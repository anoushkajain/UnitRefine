"""unitrefine-apply and unitrefine-download-models: applying models without the GUI, e.g. on a computing cluster."""

import shutil
import subprocess
import sys
from pathlib import Path

import pandas as pd
import pytest
import spikeinterface.full as si

from unitrefine import cli
from unitrefine.models_hub import HFH_PATH_FILE
from unitrefine.predictions import get_predictions, predictions_table

LAYER1, LAYER2, SUA_VS_ALL = "test_model_noise_vs_neural", "test_model_sua_vs_mua", "test_model_sua_vs_all"


def models(project):
    return Path(project) / "models"


def the_gui_predictions(analyzer_path, mode, model_folders):
    """The predictions table the GUI saves (see apply_to_analyzers), with unit ids as text as read from csv."""
    analyzer = si.load_sorting_analyzer(analyzer_path, load_extensions=False)
    table = predictions_table(analyzer.unit_ids, *get_predictions(analyzer, mode, model_folders)[:2])
    table.index = table.index.astype(str)
    return table


def read_predictions(path):
    return pd.read_csv(path, index_col="unit_id", dtype={"unit_id": str})


def test_apply_gives_the_gui_predictions(trained_project, recordings, tmp_path, capsys):
    output = tmp_path / "predictions"
    assert cli.apply_main([str(recordings["new"]), "--model", str(models(trained_project) / SUA_VS_ALL),
                           "--model", str(models(trained_project) / LAYER1), "--output", str(output)]) == 0
    saved = read_predictions(output / f"new_analyzer__labels_from_{SUA_VS_ALL}+{LAYER1}.csv")
    expected = the_gui_predictions(recordings["new"], "binary", [models(trained_project) / name for name in [SUA_VS_ALL, LAYER1]])
    pd.testing.assert_frame_equal(saved, expected, check_dtype=False)
    assert "Labelled 1/1 analyzers" in capsys.readouterr().out


def test_a_cascade_is_applied_in_the_right_order(trained_project, recordings, tmp_path):
    """The layers can be given in either order: the noise vs neural model is always applied first."""
    output = tmp_path / "predictions"
    assert cli.apply_main([str(recordings["new"]), "--model", str(models(trained_project) / LAYER2),
                           "--model", str(models(trained_project) / LAYER1), "--cascade", "--output", str(output)]) == 0
    saved = read_predictions(output / f"new_analyzer__labels_from_cascade__{LAYER1}__{LAYER2}.csv")
    expected = the_gui_predictions(recordings["new"], "cascade", [models(trained_project) / name for name in [LAYER1, LAYER2]])
    pd.testing.assert_frame_equal(saved, expected, check_dtype=False)
    assert set(saved["cascade_model_label"]) <= {"good", "MUA", "noise"}


def test_a_cascade_needs_a_layer_1_and_a_layer_2_model(trained_project, recordings, tmp_path, capsys):
    assert cli.apply_main([str(recordings["new"]), "--model", str(models(trained_project) / SUA_VS_ALL),
                           "--model", str(models(trained_project) / LAYER1), "--cascade", "--output", str(tmp_path)]) == 2
    assert "A cascade needs a noise vs neural model and a SUA vs MUA model" in capsys.readouterr().err


@pytest.fixture
def same_named_analyzers(recordings, tmp_path):
    """Two analyzers with the same folder name in different session folders, as sorters often save them."""
    analyzers = []
    for session in ["session_a", "session_b"]:
        analyzers.append(tmp_path / "data" / session / "sorting_analyzer")
        shutil.copytree(recordings["new"], analyzers[-1])
    return analyzers


def test_analyzers_are_found_inside_folders_and_listed_in_text_files(same_named_analyzers, tmp_path, capsys):
    assert cli.collect_analyzers([tmp_path / "data"]) == same_named_analyzers
    analyzer_list = tmp_path / "analyzers.txt"
    analyzer_list.write_text(f"# my recordings\n{same_named_analyzers[1]}\n\n{same_named_analyzers[0]}\n")
    assert cli.collect_analyzers([analyzer_list]) == [same_named_analyzers[1], same_named_analyzers[0]]

    assert cli.apply_main([str(analyzer_list), "--model", "unused", "--list"]) == 0
    printed = capsys.readouterr().out
    assert f"0\t{same_named_analyzers[1]}" in printed and "--array=0-1" in printed


def test_each_job_of_an_array_labels_one_analyzer_with_its_own_file_name(trained_project, same_named_analyzers, tmp_path):
    output = tmp_path / "predictions"
    for task_index in [1, 0]:
        assert cli.apply_main([str(tmp_path / "data"), "--model", str(models(trained_project) / SUA_VS_ALL),
                               "--task_index", str(task_index), "--output", str(output)]) == 0
    assert sorted(path.name for path in output.iterdir()) == [
        f"session_a_sorting_analyzer__labels_from_{SUA_VS_ALL}.csv",
        f"session_b_sorting_analyzer__labels_from_{SUA_VS_ALL}.csv",
    ]


def test_a_task_index_out_of_range_is_reported(trained_project, same_named_analyzers, tmp_path, capsys):
    assert cli.apply_main([str(tmp_path / "data"), "--model", str(models(trained_project) / SUA_VS_ALL),
                           "--task_index", "2", "--output", str(tmp_path / "predictions")]) == 2
    assert "there are 2 analyzers, numbered 0 to 1" in capsys.readouterr().err


def test_an_analyzer_without_the_metrics_is_reported_and_the_others_are_labelled(trained_project, recordings, tmp_path, capsys):
    recording, sorting = si.generate_ground_truth_recording(durations=[5.0], num_units=4, seed=3)
    no_metrics = tmp_path / "no_metrics"
    si.create_sorting_analyzer(sorting, recording, folder=no_metrics, format="binary_folder")
    output = tmp_path / "predictions"
    assert cli.apply_main([str(no_metrics), str(recordings["new"]), "--model", str(models(trained_project) / SUA_VS_ALL),
                           "--output", str(output)]) == 1
    assert [path.name for path in output.iterdir()] == [f"new_analyzer__labels_from_{SUA_VS_ALL}.csv"]
    assert f"Not labelled: {no_metrics}" in capsys.readouterr().err


def analyzer_without_metrics(folder, with_recording=True):
    recording, sorting = si.generate_ground_truth_recording(durations=[20.0], num_units=6, seed=4)
    if with_recording:
        # saved, so the analyzer knows where its recording is (a generated one isn't saved with it)
        recording = recording.save(folder=folder.parent / f"{folder.name}_recording")
    analyzer = si.create_sorting_analyzer(sorting, recording, folder=folder, format="binary_folder", sparse=True)
    analyzer.compute(["random_spikes", "templates"])
    return folder


def test_the_missing_metrics_can_be_computed_first(trained_project, tmp_path, capsys):
    from unitrefine.metrics import metrics_to_compute, required_metric_columns

    analyzer_folder = analyzer_without_metrics(tmp_path / "no_metrics")
    model = models(trained_project) / SUA_VS_ALL
    output = tmp_path / "predictions"
    assert cli.apply_main([str(analyzer_folder), "--model", str(model), "--output", str(output)]) == 1
    assert "add --compute_missing_metrics" in capsys.readouterr().err

    assert cli.apply_main([str(analyzer_folder), "--model", str(model), "--output", str(output),
                           "--compute_missing_metrics"]) == 0
    assert len(read_predictions(output / f"no_metrics__labels_from_{SUA_VS_ALL}.csv")) == 6
    # the metrics are saved in the analyzer, so they are only computed once
    analyzer = si.load_sorting_analyzer(analyzer_folder, load_extensions=False)
    assert metrics_to_compute(analyzer, [model]) == {}
    assert required_metric_columns([model]) <= set(analyzer.get_extension("quality_metrics").get_data().columns) | set(
        analyzer.get_extension("template_metrics").get_data().columns)


def test_metrics_which_need_the_recording_are_reported_without_it(trained_project, tmp_path, capsys):
    analyzer_folder = analyzer_without_metrics(tmp_path / "no_recording", with_recording=False)
    assert cli.apply_main([str(analyzer_folder), "--model", str(models(trained_project) / SUA_VS_ALL),
                           "--output", str(tmp_path / "predictions"), "--compute_missing_metrics"]) == 1
    assert "The analyzer has no recording, which is needed to compute noise_levels" in capsys.readouterr().err


def test_only_the_missing_metrics_are_computed(trained_project, recordings, tmp_path):
    """The metrics an analyzer has are kept: only the ones the models need which it doesn't have are added."""
    import numpy as np
    from unitrefine.import_labels import compute_all_metrics
    from unitrefine.metrics import compute_missing_metrics, required_metric_columns

    analyzer =si.load_sorting_analyzer(shutil.copytree(recordings["new"], tmp_path / "analyzer"), load_extensions=False)
    analyzer.delete_extension("template_metrics")
    analyzer.compute("template_metrics", metric_names=["peak_to_trough_duration"])
    before = compute_all_metrics(analyzer)
    computed = compute_missing_metrics(analyzer, [models(trained_project) / SUA_VS_ALL])
    assert list(computed) == ["template_metrics"] and "peak_to_trough_duration" not in computed["template_metrics"]
    after = compute_all_metrics(si.load_sorting_analyzer(tmp_path / "analyzer", load_extensions=False))
    assert required_metric_columns([models(trained_project) / SUA_VS_ALL]) <= set(after.columns)
    for column in before.columns:
        assert np.allclose(before[column].astype(float), after[column].astype(float), equal_nan=True), column


def test_metrics_with_the_names_from_before_spikeinterface_0_103_2_count(trained_project, recordings, monkeypatch):
    """Analyzers made with an older SpikeInterface name some template metrics differently (e.g. peak_to_valley for
    peak_to_trough_duration). The models accept these, so they aren't computed again."""
    from unitrefine import metrics
    from unitrefine.import_labels import compute_all_metrics
    from unitrefine.predictions import RENAMED_TEMPLATE_METRICS

    analyzer = si.load_sorting_analyzer(recordings["new"], load_extensions=False)
    old_names = {new_name: old_name for old_name, (new_name, _) in RENAMED_TEMPLATE_METRICS.items()}
    with_old_names = compute_all_metrics(analyzer).rename(columns=old_names)
    assert "peak_to_valley" in with_old_names.columns
    monkeypatch.setattr(metrics, "compute_all_metrics", lambda analyzer: with_old_names)
    assert metrics.metrics_to_compute(analyzer, [models(trained_project) / SUA_VS_ALL]) == {}


def test_extensions_deleted_by_computing_waveforms_are_computed_again(tmp_path):
    """The PCA metrics need waveforms, and computing them makes SpikeInterface delete the templates and the
    template metrics: these come back, with the same values."""
    import numpy as np
    from unitrefine.metrics import compute_keeping_children

    analyzer = si.load_sorting_analyzer(analyzer_without_metrics(tmp_path / "analyzer"), load_extensions=False)
    analyzer.compute("template_metrics", include_multi_channel_metrics=True)
    before = analyzer.get_extension("template_metrics").get_data()
    compute_keeping_children(analyzer, "waveforms")
    analyzer = si.load_sorting_analyzer(tmp_path / "analyzer", load_extensions=False)
    assert analyzer.has_extension("waveforms") and analyzer.has_extension("templates")
    after = analyzer.get_extension("template_metrics").get_data()
    assert list(after.columns) == list(before.columns)
    assert np.allclose(before.astype(float), after.astype(float), equal_nan=True)


def test_a_missing_model_or_analyzer_is_a_message_not_a_traceback(trained_project, recordings, tmp_path, capsys):
    assert cli.apply_main([str(recordings["new"]), "--model", str(tmp_path / "no_model")]) == 2
    assert "was not found" in capsys.readouterr().err
    assert cli.apply_main([str(tmp_path / "nothing_here"), "--model", str(models(trained_project) / SUA_VS_ALL)]) == 2
    assert "does not exist" in capsys.readouterr().err


@pytest.fixture
def fake_hub(trained_project, monkeypatch):
    """The Hugging Face Hub, with the trained models as its repos: returns the repo ids downloaded."""
    downloads = []

    def download_model(repo_id, model_directory):
        downloads.append(repo_id)
        shutil.copytree(models(trained_project) / repo_id.split("/")[-1], model_directory)
        (Path(model_directory) / HFH_PATH_FILE).write_text(repo_id)

    monkeypatch.setattr(cli, "download_model", download_model)
    return downloads


def test_a_hugging_face_model_is_downloaded_once(fake_hub, recordings, tmp_path):
    models_folder = tmp_path / "models"
    for _ in range(2):
        assert cli.apply_main([str(recordings["new"]), "--model", f"someone/{SUA_VS_ALL}", "--models_folder",
                               str(models_folder), "--output", str(tmp_path / "predictions")]) == 0
    assert fake_hub == [f"someone/{SUA_VS_ALL}"]
    # only the model is left in the models folder, not the folder it was downloaded to
    assert [path.name for path in models_folder.iterdir()] == [SUA_VS_ALL]


def test_models_downloaded_beforehand_are_used_without_internet(fake_hub, recordings, tmp_path, monkeypatch):
    models_folder = tmp_path / "models"
    assert cli.download_main([f"someone/{LAYER1}", f"someone/{LAYER2}", "--models_folder", str(models_folder)]) == 0

    def no_internet(repo_id, model_directory):
        raise ConnectionError("no internet")

    monkeypatch.setattr(cli, "download_model", no_internet)
    # the models folder can also be given as an environment variable, e.g. in a job script
    monkeypatch.setenv(cli.MODELS_FOLDER_VARIABLE, str(models_folder))
    assert cli.apply_main([str(recordings["new"]), "--model", f"someone/{LAYER1}", "--model", f"someone/{LAYER2}",
                           "--cascade", "--output", str(tmp_path / "predictions")]) == 0


def test_a_model_which_cant_be_downloaded_says_how_to_download_it_first(fake_hub, recordings, tmp_path, monkeypatch, capsys):
    def no_internet(repo_id, model_directory):
        Path(model_directory).mkdir()
        raise ConnectionError("no internet")

    monkeypatch.setattr(cli, "download_model", no_internet)
    assert cli.apply_main([str(recordings["new"]), "--model", f"someone/{SUA_VS_ALL}", "--models_folder",
                           str(tmp_path / "models")]) == 2
    assert f"unitrefine-download-models someone/{SUA_VS_ALL}" in capsys.readouterr().err


def test_a_model_downloaded_by_another_job_at_the_same_time_is_used(fake_hub, tmp_path, trained_project, monkeypatch):
    models_folder = tmp_path / "models"
    real_download = cli.download_model

    def download_while_another_job_finishes_first(repo_id, model_directory):
        real_download(repo_id, model_directory)
        shutil.copytree(model_directory, models_folder / SUA_VS_ALL)

    monkeypatch.setattr(cli, "download_model", download_while_another_job_finishes_first)
    assert cli.resolve_model(f"someone/{SUA_VS_ALL}", models_folder) == models_folder / SUA_VS_ALL
    assert [path.name for path in models_folder.iterdir()] == [SUA_VS_ALL]


def test_another_model_with_the_same_folder_name_is_not_used(fake_hub, tmp_path):
    cli.resolve_model(f"someone/{SUA_VS_ALL}", tmp_path)
    with pytest.raises(cli.UsageError, match="was not downloaded from other/"):
        cli.resolve_model(f"other/{SUA_VS_ALL}", tmp_path)


def test_download_without_repo_ids_lists_the_published_models(capsys):
    assert cli.download_main([]) == 0
    assert "SpikeInterface/UnitRefine_sua_mua_classifier" in capsys.readouterr().out


def test_apply_runs_without_qt(trained_project, recordings, tmp_path):
    """On a cluster there is often no screen, and Qt may not even load: the command doesn't import it."""
    output = tmp_path / "predictions"
    script = (
        "import sys\n"
        "from unitrefine.cli import apply_main\n"
        f"status = apply_main([{str(recordings['new'])!r}, '--model', {str(models(trained_project) / SUA_VS_ALL)!r}, "
        f"'--output', {str(output)!r}])\n"
        "qt = [name for name in sys.modules if name.startswith(('PyQt5', 'pyqtgraph'))]\n"
        "print('QT_MODULES', qt)\n"
        "sys.exit(status if len(qt) == 0 else 3)\n"
    )
    result = subprocess.run([sys.executable, "-c", script], capture_output=True, text=True)
    assert result.returncode == 0, result.stdout[-2000:] + result.stderr[-2000:]
    assert (output / f"new_analyzer__labels_from_{SUA_VS_ALL}.csv").is_file()


def test_the_commands_are_installed():
    """`uv sync` and `pip install` make unitrefine-apply and unitrefine-download-models (see pyproject.toml)."""
    scripts = Path(sys.executable).parent
    for command in ["unitrefine-apply", "unitrefine-download-models"]:
        assert any((scripts / f"{command}{suffix}").is_file() for suffix in ["", ".exe"]), f"{command} is not installed"
