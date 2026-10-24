"""
Translate a manual curation done in phy back into the SpikeInterface curation format.

The phy folder must come from `spikeinterface.exporters.export_to_phy`. phy only stores the end
result of a curation (the final cluster of every spike), so the curation is reconstructed as an
equivalent sequence of SpikeInterface steps: splits, then merges, then labels on the final units.
The result is the same as doing the curation in SpikeInterface-GUI.
"""

from pathlib import Path
import warnings

import numpy as np
import pandas as pd
from spikeinterface.core import SortingAnalyzer
from spikeinterface.curation import apply_curation
from spikeinterface.curation.curation_model import Curation, SequentialCuration

# phy group / UnitRefine labels -> SpikeInterface-GUI quality labels
QUALITY_LABELS = {"good": "good", "sua": "good", "mua": "MUA", "noise": "noise"}
QUALITY_OPTIONS = ["good", "MUA", "noise"]
UNLABELED = {"", "unsorted", "nan"}

# cluster_*.tsv files that phy or export_to_phy write but that are not curation labels
NON_LABEL_FILES = {"cluster_group.tsv", "cluster_info.tsv", "cluster_si_unit_ids.tsv", "cluster_channel_group.tsv"}


class HardMergeRequired(RuntimeError):
    """Raised when phy merges can only be applied as hard merges, which need the raw recording."""


def phy_to_curation(phy_folder, sorting_or_analyzer):
    """
    Build a SpikeInterface curation from the changes made in phy.

    Parameters
    ----------
    phy_folder : str | Path
        Folder created by `export_to_phy` and curated in phy.
    sorting_or_analyzer : BaseSorting | SortingAnalyzer
        The sorting or analyzer that was exported to phy.

    Returns
    -------
    curation : SequentialCuration
        Curation that can be applied with `spikeinterface.curation.apply_curation`.
    cluster_to_unit : dict
        Maps each final phy cluster id to the unit id it gets after the curation.
    """
    phy_folder = Path(phy_folder)
    sorting = getattr(sorting_or_analyzer, "sorting", sorting_or_analyzer)
    if sorting.get_num_segments() != 1:
        raise ValueError("export_to_phy only exports the first segment, so only single-segment sortings are supported")

    unit_ids = sorting.unit_ids.tolist()
    original = np.load(phy_folder / "spike_templates.npy").ravel()  # unit index per spike, never changed by phy
    final = np.load(phy_folder / "spike_clusters.npy").ravel()  # cluster per spike after curation

    counts = np.bincount(original, minlength=len(unit_ids))
    if len(final) != len(original) or not np.array_equal(counts, sorting.count_num_spikes_per_unit(outputs="array")):
        raise ValueError(f"{phy_folder} does not match the spike counts of this sorting - was it exported from it?")

    # final clusters of each unit's spikes, in spike train order
    order = np.argsort(original, kind="stable")
    final_per_unit = np.split(final[order], np.cumsum(counts)[:-1])

    whole_units = {}  # final cluster -> units that went into it completely
    split_parts = {}  # unit -> [(final cluster, spike indices within the unit)]
    for unit_id, unit_clusters in zip(unit_ids, final_per_unit):
        clusters = np.unique(unit_clusters)
        if len(clusters) == 1:
            whole_units.setdefault(int(clusters[0]), []).append(unit_id)
        elif len(clusters) > 1:
            split_parts[unit_id] = [(int(c), np.flatnonzero(unit_clusters == c).tolist()) for c in clusters]

    split_pieces = {}  # final cluster -> number of split pieces that went into it
    for parts in split_parts.values():
        for cluster, _ in parts:
            split_pieces[cluster] = split_pieces.get(cluster, 0) + 1

    used_ids = set(unit_ids)
    use_str = len(unit_ids) > 0 and isinstance(unit_ids[0], str)

    def take_id(preferred=None):
        new_id = None if preferred is None else (str(preferred) if use_str else preferred)
        if new_id is None or new_id in used_ids:
            new_id = max((int(u) for u in used_ids if str(u).isdigit()), default=-1) + 1
            new_id = str(new_id) if use_str else new_id
        used_ids.add(new_id)
        return new_id

    # unchanged units keep their id, new units get their phy cluster id if it is free
    cluster_to_unit = {}
    for cluster in sorted(set(whole_units) | set(split_pieces)):
        units = whole_units.get(cluster, [])
        if len(units) == 1 and cluster not in split_pieces:
            cluster_to_unit[cluster] = units[0]
        else:
            cluster_to_unit[cluster] = take_id(cluster)

    # step 1: splits. A piece that is not merged afterwards directly gets its final id
    splits = []
    for unit_id, parts in split_parts.items():
        new_ids = []
        for cluster, _ in parts:
            alone = cluster not in whole_units and split_pieces[cluster] == 1
            new_ids.append(cluster_to_unit[cluster] if alone else take_id())
        splits.append(dict(unit_id=unit_id, mode="indices", indices=[i for _, i in parts], new_unit_ids=new_ids))

    # step 2: merges of all units and pieces that ended up in the same cluster
    pieces = {cluster: list(units) for cluster, units in whole_units.items()}
    for split in splits:
        for (cluster, _), new_id in zip(split_parts[split["unit_id"]], split["new_unit_ids"]):
            pieces.setdefault(cluster, []).append(new_id)
    merges = [
        dict(unit_ids=ids, new_unit_id=cluster_to_unit[cluster]) for cluster, ids in pieces.items() if len(ids) > 1
    ]

    split_ids = [u for u in unit_ids if u not in split_parts] + [i for s in splits for i in s["new_unit_ids"]]
    merged = {i for m in merges for i in m["unit_ids"]}
    final_ids = [u for u in split_ids if u not in merged] + [m["new_unit_id"] for m in merges]

    # step 3: labels on the final units. phy copies custom columns to split/merged clusters, so for
    # new units only the curator's quality label is kept
    label_definitions, labels = _read_phy_labels(phy_folder)
    manual_labels = {}
    unknown = set()
    for name, values in labels.items():
        for cluster, value in values.items():
            if cluster not in cluster_to_unit:
                unknown.add(cluster)
                continue
            unit_id = cluster_to_unit[cluster]
            if name != "quality" and unit_id not in unit_ids:
                continue
            manual_labels.setdefault(unit_id, {"unit_id": unit_id, "labels": {}})["labels"][name] = [value]
    if unknown:
        warnings.warn(f"Ignored labels of {len(unknown)} cluster ids that no longer exist in {phy_folder}")

    steps = []
    if splits:
        steps.append(dict(format_version="2", unit_ids=unit_ids, splits=splits))
    if merges:
        steps.append(dict(format_version="2", unit_ids=split_ids, merges=merges))
    steps.append(
        dict(
            format_version="2",
            unit_ids=final_ids,
            label_definitions=label_definitions,
            manual_labels=list(manual_labels.values()),
        )
    )
    return SequentialCuration(curation_steps=steps), cluster_to_unit


def apply_phy_curation(sorting_or_analyzer, phy_folder, allow_hard_merge=False, **apply_curation_kwargs):
    """
    Apply a phy curation to the sorting or analyzer that was exported to phy.

    Unit ids that phy did not change are kept, units created in phy get their phy cluster id
    when it is free. The result is checked spike by spike against the phy folder.
    Metrics are never taken from phy: the analyzer extensions are updated by SpikeInterface.

    Merges are done as fast soft merges, which do not need the raw recording. Units that share too
    few channels can only be merged with a hard merge, which recomputes the merged units from the
    raw recording. This is only done with `allow_hard_merge=True`.

    Parameters
    ----------
    sorting_or_analyzer : BaseSorting | SortingAnalyzer
        The sorting or analyzer that was exported to phy.
    phy_folder : str | Path
        Folder created by `export_to_phy` and curated in phy.
    allow_hard_merge : bool, default: False
        If False, raise HardMergeRequired when some merges need a hard merge. If True, do these
        merges as hard merges. The raw recording must then be attached to the analyzer.
    **apply_curation_kwargs
        Passed to `spikeinterface.curation.apply_curation`, e.g. `sparsity_overlap` or job kwargs.

    Returns
    -------
    curated : BaseSorting | SortingAnalyzer
        The curated sorting or analyzer (in memory).
    """
    curation, cluster_to_unit = phy_to_curation(phy_folder, sorting_or_analyzer)
    steps = list(curation.curation_steps)

    hard_merges = _merges_needing_hard_mode(
        sorting_or_analyzer, steps, apply_curation_kwargs.get("sparsity_overlap", 0.75)
    )
    if hard_merges:
        merge_list = ", ".join(f"{m.unit_ids} -> {m.new_unit_id}" for m in hard_merges)
        if not allow_hard_merge:
            raise HardMergeRequired(
                f"{len(hard_merges)} merges cannot be done as a fast soft merge because the units share too few "
                f"channels: {merge_list}. These merges need a hard merge, which requires the raw recording and will "
                "likely take a long time to run. Check that the merges are intended, attach the recording with "
                "`analyzer.set_temporary_recording(recording)` and rerun with `allow_hard_merge=True`."
            )
        if not (sorting_or_analyzer.has_recording() or sorting_or_analyzer.has_temporary_recording()):
            raise HardMergeRequired(
                f"Hard merges of {merge_list} need the raw recording, but the analyzer has none. "
                "Attach it with `analyzer.set_temporary_recording(recording)`."
            )
        steps = _separate_hard_merges(steps, hard_merges)

    # steps are applied one by one: apply_curation (spikeinterface 0.105) wraps job kwargs of a
    # SequentialCuration in a `job_kwargs` argument, which breaks hard merges
    curated = sorting_or_analyzer
    for step in steps:
        kwargs = dict(apply_curation_kwargs)
        if hard_merges and step.merges and step.merges[0] in hard_merges:
            kwargs["merging_mode"] = "hard"
        curated = apply_curation(curated, step, **kwargs)

    sorting = getattr(curated, "sorting", curated)
    spike_times = np.load(Path(phy_folder) / "spike_times.npy").ravel()
    final = np.load(Path(phy_folder) / "spike_clusters.npy").ravel()
    order = np.argsort(final, kind="stable")
    clusters, starts = np.unique(final[order], return_index=True)
    mismatched = [
        cluster
        for cluster, cluster_times in zip(clusters, np.split(spike_times[order], starts[1:]))
        if not np.array_equal(sorting.get_unit_spike_train(cluster_to_unit[int(cluster)]), cluster_times)
    ]
    if mismatched:
        raise RuntimeError(f"{len(mismatched)} curated units do not match their phy clusters, e.g. {mismatched[:5]}")
    return curated


def _merges_needing_hard_mode(sorting_or_analyzer, steps, sparsity_overlap):
    """Return the merges whose units overlap too little for a soft merge."""
    if not isinstance(sorting_or_analyzer, SortingAnalyzer) or sorting_or_analyzer.sparsity is None:
        return []
    # split pieces keep the channels of the unit they come from
    parents = {}
    for step in steps:
        for split in step.splits:
            parents.update({new_id: split.unit_id for new_id in split.new_unit_ids})
    merges = [merge for step in steps for merge in step.merges]
    groups = [[parents.get(u, u) for u in merge.unit_ids] for merge in merges]
    mergeable = sorting_or_analyzer.are_units_mergeable(groups, sparsity_overlap=sparsity_overlap)
    return [merge for merge, group in zip(merges, groups) if not mergeable[tuple(group)]]


def _separate_hard_merges(steps, hard_merges):
    """Split the merge step into a step with the soft merges and a step with the hard merges."""
    new_steps = []
    for step in steps:
        if not step.merges:
            new_steps.append(step)
            continue
        soft = [m for m in step.merges if m not in hard_merges]
        unit_ids = step.unit_ids
        if soft:
            soft_step = Curation(format_version="2", unit_ids=unit_ids, merges=soft)
            new_steps.append(soft_step)
            unit_ids = soft_step.get_final_ids_from_new_unit_ids()
        new_steps.append(Curation(format_version="2", unit_ids=unit_ids, merges=hard_merges))
    return new_steps


def _read_phy_labels(phy_folder):
    """Return SpikeInterface label definitions and {label name: Series(cluster id -> label)} from a phy folder."""
    label_definitions = {"quality": dict(name="quality", label_options=QUALITY_OPTIONS, exclusive=True)}
    labels = {}

    group_file = phy_folder / "cluster_group.tsv"
    if group_file.is_file():
        group = pd.read_csv(group_file, sep="\t", index_col="cluster_id").iloc[:, 0]
        quality = _to_quality_labels(group)
        if quality is None:
            raise ValueError(f"Unknown phy group labels in {group_file}: {sorted(set(group.dropna()))}")
        labels["quality"] = quality

    for tsv in sorted(phy_folder.glob("cluster_*.tsv")):
        if tsv.name in NON_LABEL_FILES:
            continue
        table = pd.read_csv(tsv, sep="\t", index_col="cluster_id")
        for name in table.columns:
            if pd.api.types.is_numeric_dtype(table[name]) or name in label_definitions:
                continue  # metrics exported by export_to_phy, or a name that is already taken
            values = _to_quality_labels(table[name])
            if values is not None:
                options = QUALITY_OPTIONS
            else:
                values = _drop_unlabeled(table[name])
                options = sorted(set(values))
            if len(options) < 2:
                warnings.warn(f"Skipped column {name} in {tsv.name}: SpikeInterface needs at least two label options")
                continue
            label_definitions[name] = dict(name=name, label_options=options, exclusive=True)
            labels[name] = values
    return label_definitions, labels


def _drop_unlabeled(values):
    values = values.dropna().astype(str).str.strip()
    return values[~values.str.lower().isin(UNLABELED)]


def _to_quality_labels(values):
    """Map labels to good/MUA/noise, or return None if some labels are not quality labels."""
    values = _drop_unlabeled(values)
    mapped = values.str.lower().map(QUALITY_LABELS)
    return mapped if mapped.notna().all() else None
