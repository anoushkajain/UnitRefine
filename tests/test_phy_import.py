"""
+ Phy output: making an analyzer from Kilosort output curated in Phy, or applying a Phy curation to the analyzer
it was exported from, and importing Phy's labels. Simon's tests in test_phy_curation.py check the merges and splits.
"""

import json
import shutil
import time

import numpy as np
import pandas as pd
import pytest
import spikeinterface.full as si
from spikeinterface.exporters import export_to_phy

from unitrefine.phy_import import (
    MADE_FROM_PHY_FILE, NeedsGain, RecordingNotFound, import_phy_folder, main, phy_folder_kind,
)

ONE_JOB = dict(n_jobs=1)


@pytest.fixture(scope="module")
def exported(tmp_path_factory):
    """An analyzer (8 units, with a recording) and a Phy folder exported from it, with the recording copied."""
    folder = tmp_path_factory.mktemp("phy")
    recording, sorting = si.generate_ground_truth_recording(durations=[20.0], num_units=8, seed=3)
    recording = recording.save(folder=folder / "recording")
    analyzer = si.create_sorting_analyzer(sorting, recording, format="binary_folder", folder=folder / "analyzer", sparse=True)
    analyzer.compute(["random_spikes", "waveforms", "templates", "noise_levels", "quality_metrics", "template_metrics"])
    export_to_phy(analyzer, folder / "exported_phy", compute_pc_features=False, compute_amplitudes=False,
                  copy_binary=True, verbose=False)
    return folder


def phy_copy(exported, tmp_path, kilosort=True):
    """A copy of the exported Phy folder; without cluster_si_unit_ids.tsv it is like Kilosort output."""
    phy_folder = tmp_path / ("kilosort4" if kilosort else "phy")
    shutil.copytree(exported / "exported_phy", phy_folder)
    # params.py points to the recording in the exported folder; the copy has its own
    params = (phy_folder / "params.py").read_text()
    (phy_folder / "params.py").write_text(params.replace(str(exported / "exported_phy"), str(phy_folder)))
    if kilosort:
        (phy_folder / "cluster_si_unit_ids.tsv").unlink()
    return phy_folder


def merge_in_phy(phy_folder, clusters_to_merge, new_cluster):
    clusters = np.load(phy_folder / "spike_clusters.npy").ravel()
    clusters[np.isin(clusters, clusters_to_merge)] = new_cluster
    np.save(phy_folder / "spike_clusters.npy", clusters[:, np.newaxis])


def label_in_phy(phy_folder, labels):
    pd.DataFrame({"cluster_id": list(labels), "group": list(labels.values())}).to_csv(
        phy_folder / "cluster_group.tsv", sep="\t", index=False)


def labels_of(result):
    labels = pd.read_csv(result["labels"])
    return dict(zip(labels["unit_id"].astype(str), labels["quality"]))


def test_the_kind_of_phy_folder(exported, tmp_path):
    assert phy_folder_kind(exported / "exported_phy") == "exported"
    assert phy_folder_kind(phy_copy(exported, tmp_path)) == "kilosort"
    with pytest.raises(ValueError, match="not a Phy folder"):
        phy_folder_kind(exported / "analyzer")


def test_kilosort_output_curated_in_phy(exported, tmp_path):
    phy_folder = phy_copy(exported, tmp_path)
    merge_in_phy(phy_folder, [0, 1], 8)
    # unsorted is not a label; clusters missing from cluster_group.tsv are kept (read_phy would leave them out)
    label_in_phy(phy_folder, {2: "good", 3: "mua", 4: "noise", 5: "unsorted", 8: "good"})

    result = import_phy_folder(phy_folder, gain_to_uV=1.0, job_kwargs=ONE_JOB)

    assert result["kind"] == "kilosort" and result["new_analyzer"]
    assert result["analyzer"] == str(tmp_path / "kilosort4_analyzer")
    analyzer = si.load_sorting_analyzer(result["analyzer"])
    assert sorted(int(unit_id) for unit_id in analyzer.unit_ids) == [2, 3, 4, 5, 6, 7, 8]
    original = si.load_sorting_analyzer(exported / "analyzer", load_extensions=False).sorting
    merged = np.sort(np.concatenate([original.get_unit_spike_train(unit_id) for unit_id in original.unit_ids[:2]]))
    assert np.array_equal(analyzer.sorting.get_unit_spike_train(analyzer.unit_ids[list(analyzer.unit_ids).index(8)]), merged)
    # the metrics the models use, for every unit
    metrics = analyzer.get_extension("quality_metrics").get_data()
    assert metrics["snr"].notna().all() and metrics["firing_rate"].notna().all()
    assert analyzer.has_extension("template_metrics") and analyzer.has_extension("principal_components")
    # and the drift and multi-channel template metrics, which the models on the Hugging Face Hub also use
    columns = set(metrics.columns) | set(analyzer.get_extension("template_metrics").get_data().columns)
    assert {"drift_ptp", "drift_std", "drift_mad", "spread", "velocity_above", "velocity_below", "exp_decay"} <= columns

    assert labels_of(result) == {"2": "good", "3": "MUA", "4": "noise", "8": "good"}
    made_from = json.loads((tmp_path / "kilosort4_analyzer" / MADE_FROM_PHY_FILE).read_text())
    assert made_from["phy_folder"] == str(phy_folder) and made_from["kind"] == "kilosort"


def test_kilosorts_own_labels_are_not_used(exported, tmp_path):
    phy_folder = phy_copy(exported, tmp_path)
    label_in_phy(phy_folder, {0: "good", 1: "mua"})
    pd.DataFrame({"cluster_id": [0, 1], "KSLabel": ["good", "mua"]}).to_csv(phy_folder / "cluster_KSLabel.tsv", sep="\t", index=False)
    assert import_phy_folder(phy_folder, gain_to_uV=1.0, job_kwargs=ONE_JOB)["num_labels"] == 0

    # labelled in Phy: different from Kilosort's labels
    label_in_phy(phy_folder, {0: "noise", 1: "mua"})
    result = import_phy_folder(phy_folder, gain_to_uV=1.0, job_kwargs=ONE_JOB, reuse=True)
    assert labels_of(result) == {"0": "noise", "1": "MUA"}


def test_a_plain_binary_recording_needs_its_gain(exported, tmp_path, capsys):
    phy_folder = phy_copy(exported, tmp_path)
    with pytest.raises(NeedsGain):
        import_phy_folder(phy_folder)
    # on the command line, the GUI gets the reason
    assert main([str(phy_folder)]) == 1
    result_line = [line for line in capsys.readouterr().out.splitlines() if line.startswith("UNITREFINE_PHY_RESULT:")][0]
    assert json.loads(result_line.split(":", 1)[1])["status"] == "needs_gain"


def test_the_recording_is_found_when_moved_with_the_phy_folder(exported, tmp_path):
    phy_folder = phy_copy(exported, tmp_path)
    params = (phy_folder / "params.py").read_text()
    (phy_folder / "params.py").write_text(params.replace(str(phy_folder), str(tmp_path / "somewhere_else")))
    assert import_phy_folder(phy_folder, gain_to_uV=1.0, job_kwargs=ONE_JOB)["new_analyzer"]

    (phy_folder / "recording.dat").rename(tmp_path / "recording_elsewhere.dat")
    with pytest.raises(RecordingNotFound):
        import_phy_folder(phy_folder, output_folder=tmp_path / "other", gain_to_uV=1.0, job_kwargs=ONE_JOB)
    result = import_phy_folder(phy_folder, output_folder=tmp_path / "other", recording_path=tmp_path / "recording_elsewhere.dat",
                               gain_to_uV=1.0, job_kwargs=ONE_JOB)
    assert result["new_analyzer"]


def test_exported_folder_with_only_labels_uses_the_analyzer(exported, tmp_path):
    phy_folder = phy_copy(exported, tmp_path, kilosort=False)
    label_in_phy(phy_folder, {0: "good", 1: "noise", 2: "unsorted"})
    result = import_phy_folder(phy_folder, analyzer_path=exported / "analyzer", job_kwargs=ONE_JOB)
    # nothing to save: the analyzer is used as it is
    assert result["analyzer"] == str(exported / "analyzer") and not result["new_analyzer"]
    assert not (tmp_path / "phy_analyzer").exists()
    assert labels_of(result) == {"0": "good", "1": "noise"}


def test_exported_folder_with_a_merge_saves_the_curated_analyzer(exported, tmp_path):
    phy_folder = phy_copy(exported, tmp_path, kilosort=False)
    merge_in_phy(phy_folder, [0, 1], 8)
    label_in_phy(phy_folder, {8: "good", 3: "mua"})
    result = import_phy_folder(phy_folder, analyzer_path=exported / "analyzer", job_kwargs=ONE_JOB)
    assert result["new_analyzer"] and result["analyzer"] == str(tmp_path / "phy_analyzer")
    curated = si.load_sorting_analyzer(result["analyzer"])
    assert curated.get_num_units() == 7
    assert curated.get_extension("quality_metrics").get_data()["snr"].notna().all()
    assert labels_of(result) == {"8": "good", "3": "MUA"}


def test_an_exported_folder_needs_its_analyzer(exported, tmp_path):
    with pytest.raises(ValueError, match="give that analyzer"):
        import_phy_folder(phy_copy(exported, tmp_path, kilosort=False), job_kwargs=ONE_JOB)


# --------------------------------------------------------------------------------------------------
# the + Phy output buttons
# --------------------------------------------------------------------------------------------------

def run_button(window, qapp, folders, for_training, monkeypatch, gain=1.0):
    """
    Clicks + Phy output in section 2 or 3.3, answering the folder dialogs with `folders` (the Phy folder first),
    and waits for the analyzer to be made. Returns the messages shown.
    """
    import PyQt5.QtWidgets as QtWidgets

    messages = []
    answers = iter(str(folder) for folder in folders)
    monkeypatch.setattr(QtWidgets.QFileDialog, "getExistingDirectory", staticmethod(lambda *args, **kwargs: next(answers)))
    monkeypatch.setattr(QtWidgets.QInputDialog, "getDouble", staticmethod(lambda *args, **kwargs: (gain, True)))
    for name in ["information", "warning"]:
        monkeypatch.setattr(QtWidgets.QMessageBox, name, staticmethod(lambda parent, title, text, *args: messages.append((title, text))))
    monkeypatch.setattr(type(window), "phy_import_options", ["--n_jobs", "1"])

    window.add_phy_output(for_training=for_training)
    end = time.time() + 600
    while window.phy_process is not None and time.time() < end:
        qapp.processEvents()
        time.sleep(0.05)
    assert window.phy_process is None, "the analyzer was not made in 10 minutes"
    return messages


def test_phy_output_in_section_2(exported, tmp_path, qapp, monkeypatch):
    from unitrefine.main import MainWindow, UnitRefineProject

    phy_folder = phy_copy(exported, tmp_path)
    label_in_phy(phy_folder, {2: "good", 3: "mua", 4: "noise"})
    project = UnitRefineProject(tmp_path / "project")
    project.save()
    window = MainWindow(project)

    messages = run_button(window, qapp, [phy_folder], True, monkeypatch)

    assert messages[-1][0] == "Phy folder added", messages
    assert [analyzer["path"] for analyzer in window.project.analyzers.values()] == [str(tmp_path / "kilosort4_analyzer")]
    labels = pd.read_csv(tmp_path / "project" / "analyzers" / "analyzer_0" / "labels.csv", dtype=str)
    assert dict(zip(labels["unit_id"], labels["quality"])) == {"2": "good", "3": "MUA", "4": "noise"}
    # ready to train on: the labelled units' metrics are saved with them
    assert len(pd.read_csv(tmp_path / "project" / "analyzers" / "analyzer_0" / "labelled_metrics.csv")) == 3
    window.close()


def test_phy_output_in_section_3(exported, tmp_path, trained_project, project_copy, qapp, monkeypatch):
    from unitrefine.main import MainWindow, load_project

    phy_folder = phy_copy(exported, tmp_path, kilosort=False)
    label_in_phy(phy_folder, {0: "good", 1: "noise"})
    folder = project_copy(trained_project)
    window = MainWindow(load_project(folder))
    selector = window.model_selector
    for row in range(selector.binary_list.count()):
        item = selector.binary_list.item(row)
        item.setCheckState(2 if item.data(256) == "test_model_sua_vs_all" else 0)
    # the analyzer Phy was exported from is found when it is in the project; here it is not, so it is asked for
    messages = run_button(window, qapp, [phy_folder, exported / "analyzer"], False, monkeypatch)

    applied = window.project.config.get("applied_analyzers", [])
    assert messages[-1][0] == "Phy folder added", messages
    assert [entry["path"] for entry in applied] == [str(exported / "analyzer")]
    labels = pd.read_csv(f"{applied[0]['output_folder']}/{applied[0]['output_name']}__labels.csv", dtype=str)
    assert dict(zip(labels["unit_id"], labels["quality"])) == {"0": "good", "1": "noise"}
    window.close()
