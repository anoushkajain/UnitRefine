"""
Import unit labels produced outside UnitRefine (Phy, SortingView, SpikeInterface curation
json, or a plain csv) and save them in the project, in the same format as labels made
with the SpikeInterface-GUI, so they can be used to train a model.

Supported files:
    - Phy: cluster_group.tsv, cluster_info.tsv, cluster_KSLabel.tsv (columns `cluster_id` + `group`/`KSLabel`)
    - SortingView curation json: {"labelsByUnit": {"<unit_id>": ["good"], ...}}
    - SpikeInterface curation json: {"manual_labels": [...]}
    - csv/tsv with columns `unit_id` and `quality`
"""

import json
from pathlib import Path

import pandas as pd
from spikeinterface.core import load_sorting_analyzer

# Maps (lowercased) external label names to the labels used by UnitRefine.
LABEL_MAP = {
    "good": "good",
    "sua": "good",
    "single": "good",
    "mua": "MUA",
    "multi": "MUA",
    "multi-unit": "MUA",
    "noise": "noise",
    "artifact": "noise",
}


def read_label_file(label_file):
    """Reads a label file and returns a DataFrame with columns `unit_id` and `raw_label`."""

    label_file = Path(label_file)

    if label_file.suffix.lower() == ".json":
        with open(label_file) as f:
            curation = json.load(f)

        rows = []
        if "labelsByUnit" in curation:
            # SortingView
            for unit_id, unit_labels in curation["labelsByUnit"].items():
                rows.append({"unit_id": unit_id, "raw_label": _first_known_label(unit_labels)})
            if len(curation.get("mergeGroups", [])) > 0:
                print("Warning: merge groups in the SortingView curation are ignored, only labels are imported.")
        elif "manual_labels" in curation:
            # SpikeInterface curation format (v1 and v2)
            for row in curation["manual_labels"]:
                unit_labels = row.get("labels", row).get("quality", [])
                rows.append({"unit_id": row["unit_id"], "raw_label": _first_known_label(unit_labels)})
        else:
            raise ValueError(f"{label_file} is not a recognised SortingView or SpikeInterface curation json.")

        return pd.DataFrame(rows, columns=["unit_id", "raw_label"])

    sep = "\t" if label_file.suffix.lower() == ".tsv" else ","
    table = pd.read_csv(label_file, sep=sep)

    unit_id_column = next((col for col in ["unit_id", "cluster_id", "id"] if col in table.columns), None)
    # Phy stores manual labels in `group`; `KSLabel` holds Kilosort's automatic labels.
    label_column = next((col for col in ["quality", "group", "KSLabel", "label"] if col in table.columns), None)

    if unit_id_column is None or label_column is None:
        raise ValueError(
            f"Could not find unit id and label columns in {label_file}. "
            f"Found columns: {list(table.columns)}"
        )

    print(f"Reading labels from column '{label_column}' and unit ids from column '{unit_id_column}'.")
    return pd.DataFrame({"unit_id": table[unit_id_column].values, "raw_label": table[label_column].values})


def _first_known_label(unit_labels):
    if isinstance(unit_labels, str):
        unit_labels = [unit_labels]
    for label in unit_labels:
        if str(label).lower() in LABEL_MAP:
            return label
    return unit_labels[0] if len(unit_labels) > 0 else None


def compute_all_metrics(analyzer):
    """Concatenates the quality and template metrics of an analyzer, indexed by unit id."""

    for extension_name in ["quality_metrics", "template_metrics"]:
        # load_extension raises if the extension was never computed
        try:
            analyzer.load_extension(extension_name)
        except Exception:
            pass

    if analyzer.has_extension("template_metrics"):
        tms = analyzer.get_extension("template_metrics").get_data()
    else:
        tms = pd.DataFrame()

    if analyzer.has_extension("quality_metrics"):
        qms = analyzer.get_extension("quality_metrics").get_data()
    else:
        qms = pd.DataFrame()

    return pd.concat([qms, tms], axis=1)


def import_labels(label_file, analyzer_path, save_folder, existing_labels=None, file_prefix=""):
    """
    Matches the labels in `label_file` to the units of the analyzer at `analyzer_path` and saves
    `labels.csv`, `labelled_metrics.csv` and `all_metrics.csv` in `save_folder`, their names starting
    with `file_prefix` (used for new analyzers, whose files share one folder).

    Imported labels replace existing labels of the same units; other existing labels are kept.
    Returns the labels DataFrame (columns `quality`, `unit_id`).
    """

    save_folder = Path(save_folder)

    raw_labels = read_label_file(label_file)

    analyzer = load_sorting_analyzer(analyzer_path, load_extensions=False)
    all_metrics = compute_all_metrics(analyzer)
    if all_metrics.empty:
        raise ValueError(
            "The analyzer has no quality or template metrics. Please compute them before importing labels."
        )

    # unit ids in label files are often stored as int or str, independent of the analyzer
    analyzer_unit_ids = {str(unit_id): unit_id for unit_id in analyzer.unit_ids}

    unit_ids, qualities = [], []
    missing_units, unknown_labels = [], set()
    for unit_id, raw_label in zip(raw_labels["unit_id"], raw_labels["raw_label"]):
        if pd.isna(raw_label):
            continue
        quality = LABEL_MAP.get(str(raw_label).strip().lower())
        if quality is None:
            unknown_labels.add(str(raw_label))
            continue
        analyzer_unit_id = analyzer_unit_ids.get(str(unit_id))
        if analyzer_unit_id is None:
            missing_units.append(unit_id)
            continue
        unit_ids.append(analyzer_unit_id)
        qualities.append(quality)

    if len(unknown_labels) > 0:
        print(f"Skipped units with labels not used by UnitRefine: {sorted(unknown_labels)}")
    if len(missing_units) > 0:
        print(f"Skipped {len(missing_units)} units not found in the analyzer: {missing_units[:20]}")
    if len(unit_ids) == 0:
        raise ValueError("No labels could be matched to units in the analyzer.")

    labels_df = pd.DataFrame({"quality": qualities, "unit_id": unit_ids})

    if existing_labels is not None and len(existing_labels) > 0:
        kept = existing_labels[["quality", "unit_id"]].copy()
        # labels.csv is re-read with pandas' inferred dtype, so map back to the analyzer's unit ids
        kept["unit_id"] = kept["unit_id"].astype(str).map(analyzer_unit_ids)
        kept = kept.dropna(subset=["unit_id"])
        kept = kept[~kept["unit_id"].isin(labels_df["unit_id"])]
        labels_df = pd.concat([kept, labels_df])

    labels_df = labels_df.sort_values("unit_id").reset_index(drop=True)
    # keep labelled_metrics in the same order as labels.csv, which TrainWindow relies on
    labelled_metrics = all_metrics.loc[labels_df["unit_id"].values]

    save_folder.mkdir(exist_ok=True, parents=True)
    all_metrics.to_csv(save_folder / f"{file_prefix}all_metrics.csv", index_label="unit_id")
    labelled_metrics.to_csv(save_folder / f"{file_prefix}labelled_metrics.csv", index_label="unit_id")
    labels_df.to_csv(save_folder / f"{file_prefix}labels.csv", index=False)

    counts = labels_df["quality"].value_counts().to_dict()
    print(f"Imported labels for {len(labels_df)}/{len(all_metrics)} units: {counts}")

    return labels_df
