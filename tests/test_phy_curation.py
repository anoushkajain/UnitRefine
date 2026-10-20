import numpy as np
import pandas as pd
import pytest
import spikeinterface.full as si
from spikeinterface.exporters import export_to_phy

from unitrefine.phy_curation import apply_phy_curation, phy_to_curation


@pytest.fixture(scope="module", params=[str, int])
def analyzer(request):
    recording, sorting = si.generate_ground_truth_recording(durations=[20.0], num_units=6, seed=0)
    sorting = sorting.rename_units([request.param(i) for i in range(6)])
    analyzer = si.create_sorting_analyzer(sorting, recording, sparse=False)
    analyzer.compute(["random_spikes", "waveforms", "templates", "noise_levels", "spike_amplitudes"])
    return analyzer


@pytest.fixture
def phy_folder(analyzer, tmp_path):
    folder = tmp_path / "phy"
    export_to_phy(analyzer, folder, compute_pc_features=False, copy_binary=False, verbose=False)
    return folder


def edit_clusters(phy_folder, edit):
    """Simulate phy: edit(clusters, units) changes the cluster of each spike in place."""
    clusters = np.load(phy_folder / "spike_clusters.npy").ravel()
    units = np.load(phy_folder / "spike_templates.npy").ravel()
    edit(clusters, units)
    np.save(phy_folder / "spike_clusters.npy", clusters[:, np.newaxis])


def write_labels(phy_folder, labels, name="group"):
    file_name = "cluster_group.tsv" if name == "group" else f"cluster_{name}.tsv"
    table = pd.DataFrame({"cluster_id": list(labels), name: list(labels.values())})
    table.to_csv(phy_folder / file_name, sep="\t", index=False)


def spikes_of(sorting, unit_ids):
    return np.sort(np.concatenate([sorting.get_unit_spike_train(as_id(sorting, u)) for u in unit_ids]))


def as_id(sorting, i):
    return str(i) if isinstance(sorting.unit_ids[0], str) else i


def ids_of(sorting_or_analyzer):
    return sorted(int(u) for u in sorting_or_analyzer.unit_ids)


def quality_of(curated):
    return {int(u): q for u, q in zip(curated.unit_ids, curated.sorting.get_property("quality"))}


def test_labels_only(analyzer, phy_folder):
    write_labels(phy_folder, {0: "good", 1: "mua", 2: "noise", 3: "unsorted", 4: "good", 5: "good"})
    curated = apply_phy_curation(analyzer, phy_folder)

    assert list(curated.unit_ids) == list(analyzer.unit_ids)
    assert quality_of(curated) == {0: "good", 1: "MUA", 2: "noise", 3: "", 4: "good", 5: "good"}


def test_merge(analyzer, phy_folder):
    def merge(clusters, units):
        clusters[np.isin(units, [0, 1])] = 6

    edit_clusters(phy_folder, merge)
    write_labels(phy_folder, {2: "good", 3: "good", 4: "mua", 5: "noise", 6: "good"})
    curated = apply_phy_curation(analyzer, phy_folder)

    assert ids_of(curated) == [2, 3, 4, 5, 6]
    assert np.array_equal(spikes_of(curated.sorting, [6]), spikes_of(analyzer.sorting, [0, 1]))
    assert quality_of(curated)[6] == "good" and quality_of(curated)[4] == "MUA"


@pytest.mark.parametrize("num_parts", [2, 3])
def test_split(analyzer, phy_folder, num_parts):
    def split(clusters, units):
        in_unit = np.flatnonzero(units == 2)
        for part, spikes in enumerate(np.array_split(in_unit, num_parts)):
            clusters[spikes] = 6 + part

    edit_clusters(phy_folder, split)
    curated = apply_phy_curation(analyzer, phy_folder)

    new_ids = list(range(6, 6 + num_parts))
    assert ids_of(curated) == [0, 1, 3, 4, 5] + new_ids
    assert np.array_equal(spikes_of(curated.sorting, new_ids), spikes_of(analyzer.sorting, [2]))


def test_split_then_merge(analyzer, phy_folder):
    # split unit 3 into 6 and 7, then merge 7 with unit 4 into 8
    def edit(clusters, units):
        in_unit = np.flatnonzero(units == 3)
        clusters[in_unit[: len(in_unit) // 2]] = 6
        clusters[in_unit[len(in_unit) // 2 :]] = 8
        clusters[units == 4] = 8

    edit_clusters(phy_folder, edit)
    write_labels(phy_folder, {0: "good", 1: "good", 2: "good", 5: "good", 6: "noise", 8: "mua"})
    curated = apply_phy_curation(analyzer, phy_folder)

    assert ids_of(curated) == [0, 1, 2, 5, 6, 8]
    assert quality_of(curated)[6] == "noise" and quality_of(curated)[8] == "MUA"

    # same result when applied to the sorting instead of the analyzer
    curated_sorting = apply_phy_curation(analyzer.sorting, phy_folder)
    assert ids_of(curated_sorting) == [0, 1, 2, 5, 6, 8]


def test_custom_columns(analyzer, phy_folder):
    write_labels(phy_folder, {0: "SUA", 1: "MUA", 2: "noise", 3: "SUA", 4: "SUA", 5: "MUA"}, name="modelLabels")
    write_labels(phy_folder, {i: f"region{i % 2}" for i in range(6)}, name="area")
    write_labels(phy_folder, {i: float(i) for i in range(6)}, name="score")  # numeric -> not a label
    curation, _ = phy_to_curation(phy_folder, analyzer)

    label_definitions = curation.curation_steps[-1].label_definitions
    assert label_definitions["modelLabels"].label_options == ["good", "MUA", "noise"]
    assert label_definitions["area"].label_options == ["region0", "region1"]
    assert "score" not in label_definitions

    curated = apply_phy_curation(analyzer, phy_folder)
    assert list(curated.sorting.get_property("modelLabels")) == ["good", "MUA", "noise", "good", "good", "MUA"]


def test_wrong_sorting(analyzer, phy_folder):
    other = si.generate_sorting(num_units=6, durations=[20.0], seed=1)
    with pytest.raises(ValueError):
        phy_to_curation(phy_folder, other)


def test_custom_columns_not_copied_to_new_units(analyzer, phy_folder):
    # phy copies custom columns to both halves of a split, only the quality label should be kept
    def split(clusters, units):
        in_unit = np.flatnonzero(units == 2)
        clusters[in_unit[: len(in_unit) // 2]] = 6
        clusters[in_unit[len(in_unit) // 2 :]] = 7

    edit_clusters(phy_folder, split)
    write_labels(phy_folder, {0: "good", 1: "good", 3: "good", 4: "good", 5: "good", 6: "noise", 7: "good"})
    write_labels(phy_folder, {0: "SUA", 1: "MUA", 3: "SUA", 4: "SUA", 5: "MUA", 6: "SUA", 7: "SUA"}, name="modelLabels")
    curated = apply_phy_curation(analyzer, phy_folder)

    model = {int(u): m for u, m in zip(curated.unit_ids, curated.sorting.get_property("modelLabels"))}
    assert model == {0: "good", 1: "MUA", 3: "good", 4: "good", 5: "MUA", 6: "", 7: ""}
    assert quality_of(curated)[6] == "noise" and quality_of(curated)[7] == "good"
