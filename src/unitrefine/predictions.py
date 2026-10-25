"""
Apply one or more models to an analyzer, either as independent binary models or as a two layer
cascade (noise vs neural, then SUA vs MUA on the neural units).
"""

import hashlib
import json
import os
import threading
import time
from pathlib import Path

import numpy as np
import pandas as pd

# Predictions are made and saved both by the buttons and by the figures, which are made in a background
# thread; this lock stops the two from making or writing the same predictions at the same time
PREDICTIONS_LOCK = threading.RLock()

# Model classes which mean the same as the GUI labels good / MUA / noise
QUALITY_ALIASES = {"good": "good", "sua": "good", "SUA": "good", "MUA": "MUA", "mua": "MUA", "noise": "noise"}


def as_model_input(metrics):
    """
    Metric values as SpikeInterface gives them to a model, when training (train_model) and when predicting:
    as 32-bit floats, with infinite values as missing. UnitRefine does the same when it uses a model itself
    (test sets, retraining, SHAP), so the model sees exactly the same values; 64-bit values can give a few
    different predictions for units close to a decision threshold.
    """
    return metrics.astype("float32").replace([np.inf, -np.inf], np.nan)


# Template metrics renamed in SpikeInterface 0.103.2, as {old name: (new name, sign flipped)}
RENAMED_TEMPLATE_METRICS = {
    "peak_to_valley": ("peak_to_trough_duration", False),
    "peak_trough_ratio": ("peak_after_to_trough_ratio", True),
    "half_width": ("trough_half_width", False),
}


def metrics_for_model(metrics, model_info):
    """
    SpikeInterface 0.103.2 renamed some template metrics (e.g. peak_to_valley is now peak_to_trough_duration)
    and inverted the sign of one. Models trained before expect the old names: this converts a metrics table
    the same way auto_label_units does (in every SpikeInterface version since), so retraining and SHAP use
    the values the model sees when predicting. Models trained with 0.103.2 or later are left as they are.
    """
    from packaging.version import parse

    if model_info is None:
        return metrics
    trained_with = (model_info.get("requirements") or {}).get("spikeinterface")
    if trained_with is None or parse(trained_with) >= parse("0.103.2"):
        return metrics
    metrics = metrics.copy()
    for old_name, (new_name, flip_sign) in RENAMED_TEMPLATE_METRICS.items():
        if new_name in metrics.columns:
            metrics = metrics.rename(columns={new_name: old_name})
            if flip_sign:
                metrics[old_name] = -1 * metrics[old_name]
    return metrics


def get_model_classes(model_folder):
    """Returns the classes predicted by a model, read from its model_info.json."""
    model_info_path = Path(model_folder) / "model_info.json"
    if not model_info_path.is_file():
        return []
    with open(model_info_path) as f:
        label_conversion = json.load(f).get("label_conversion", {})
    return [label_conversion[key] for key in sorted(label_conversion, key=int)]


def is_layer1_model(classes):
    """A noise vs neural model: two classes, one of them noise. Models downloaded from
    the Hugging Face Hub call the neural class 'good'."""
    return len(classes) == 2 and "noise" in classes


def is_layer2_model(classes):
    """A SUA vs MUA model: two classes which mean good and MUA."""
    return len(classes) == 2 and set(QUALITY_ALIASES.get(c) for c in classes) == {"good", "MUA"}


def map_quality_to_model_class(quality, model_classes):
    """
    Maps a curated label (good, MUA, noise) to one of a model's classes, e.g. MUA -> neural for a
    noise vs neural model. Returns None if the label is not used by the model, e.g. noise for a
    SUA vs MUA model.
    """
    model_classes = list(model_classes)
    # collapse good + MUA into neural for noise vs neural models. The neural class is called
    # "neural" in models trained here, and "good" in models downloaded from the Hugging Face Hub
    if len(model_classes) == 2 and "noise" in model_classes and quality in ("good", "MUA"):
        return [model_class for model_class in model_classes if model_class != "noise"][0]
    # label already supported by the model
    if quality in model_classes:
        return quality
    # same label with another name, e.g. good for a model trained with sua/mua
    for model_class in model_classes:
        if QUALITY_ALIASES.get(model_class) == quality:
            return model_class
    # collapse MUA + noise into non-SUA if model uses that convention
    if "not-sua" in model_classes and quality in ("MUA", "noise"):
        return "not-sua"
    return None


def selection_key(mode, model_names):
    """
    Name used for files which belong to a model selection, e.g. relabelled_units_{key}.csv.
    A single binary model keeps its own name, so files made before cascades existed still work.
    """
    if mode == "binary" and len(model_names) == 1:
        return model_names[0]
    if mode == "cascade":
        key = f"cascade__{model_names[0]}__{model_names[1]}"
    else:
        key = "+".join(model_names)
    if len(key) > 100:
        # keep file paths short enough for Windows
        short_hash = hashlib.md5(key.encode()).hexdigest()[:8]
        key = f"{mode}__{model_names[0][:40]}_and_{len(model_names) - 1}_more_{short_hash}"
    return key


def label_units_function():
    """
    SpikeInterface's function which applies a model to an analyzer: model_based_label_units, called
    auto_label_units before SpikeInterface 0.104 (0.105 only has the new name).
    """
    try:
        from spikeinterface.curation import model_based_label_units
        return model_based_label_units
    except ImportError:
        from spikeinterface.curation import auto_label_units
        return auto_label_units


def predict(sorting_analyzer, model_folder):
    # the predictions are saved by UnitRefine (csv files), not as properties of the analyzer's units: that would write
    # to the analyzer, which fails for one which can't be written to, e.g. on s3
    decisions = label_units_function()(sorting_analyzer=sorting_analyzer, model_folder=model_folder, trust_model=True,
                                       set_predictions_as_properties=False)
    return decisions.loc[sorting_analyzer.unit_ids]


def get_predictions(sorting_analyzer, mode, model_folders):
    """
    Returns
    -------
    label_columns : dict
        {column name: pd.Series of labels indexed by unit id}. Each column is named after the
        model it comes from. Units a model was not applied to are missing from its Series.
    confidence_columns : dict
        {column name: np.array of probabilities, aligned with sorting_analyzer.unit_ids}
    selection_confidence : np.array
        One confidence per unit for the whole selection, used to find the least confident units.
        Binary: the lowest confidence over the models. Cascade: p(layer 1) x p(layer 2) for
        neural units, p(layer 1) for noise units.
    """

    model_folders = [Path(model_folder) for model_folder in model_folders]
    label_columns, confidence_columns = {}, {}

    if mode == "binary":
        confidences = []
        for model_folder in model_folders:
            decisions = predict(sorting_analyzer, model_folder)
            label_columns[f"{model_folder.name}_model_label"] = decisions["prediction"].astype(str)
            confidence_columns[f"{model_folder.name}_confidence"] = decisions["probability"].values
            confidences.append(decisions["probability"].values)
        selection_confidence = np.min(confidences, axis=0)
        return label_columns, confidence_columns, selection_confidence

    layer1_folder, layer2_folder = model_folders
    layer1 = predict(sorting_analyzer, layer1_folder)
    layer2 = predict(sorting_analyzer, layer2_folder)

    is_noise = (layer1["prediction"].astype(str) == "noise").values
    layer2_labels = layer2["prediction"].astype(str).map(QUALITY_ALIASES)

    cascade_labels = pd.Series(np.where(is_noise, "noise", layer2_labels.values), index=layer1.index)
    p1 = layer1["probability"].values
    p2 = layer2["probability"].values
    cascade_confidence = np.where(is_noise, p1, p1 * p2)

    layer1_name = f"L1_{layer1_folder.name}"
    layer2_name = f"L2_{layer2_folder.name}"

    label_columns["cascade_model_label"] = cascade_labels
    label_columns[f"{layer1_name}_model_label"] = layer1["prediction"].astype(str)
    # layer 2 is only applied to the units which layer 1 calls neural
    label_columns[f"{layer2_name}_model_label"] = layer2["prediction"].astype(str)[~is_noise]

    confidence_columns["cascade_confidence"] = cascade_confidence
    confidence_columns[f"{layer1_name}_confidence"] = p1
    confidence_columns[f"{layer2_name}_confidence"] = np.where(is_noise, np.nan, p2)

    return label_columns, confidence_columns, cascade_confidence


def predictions_table(unit_ids, label_columns, confidence_columns):
    """All label and confidence columns in one DataFrame indexed by unit id, as saved to csv."""
    table = pd.DataFrame(index=unit_ids)
    for column_name, labels in label_columns.items():
        table[column_name] = labels
    for column_name, confidences in confidence_columns.items():
        table[column_name] = confidences
    table.index.name = "unit_id"
    return table


def save_csv_atomic(table, path, **to_csv_kwargs):
    """
    Writes a csv to a temporary file and then moves it into place, so a file which is read at the same
    time (by the figures, or the SpikeInterface-GUI process) is never half written.
    """
    path = Path(path)
    # a short name, so the path doesn't become longer than Windows allows (260 characters) when the name is long
    temporary_path = path.with_name(f".{os.getpid()}-{threading.get_ident()}.tmp")
    table.to_csv(temporary_path, **to_csv_kwargs)
    for attempt in range(20):
        try:
            os.replace(temporary_path, path)
            return
        except PermissionError:
            # on Windows, a file can't be replaced while another program has it open
            time.sleep(0.05)
    os.replace(temporary_path, path)


def predictions_file_name(prefix, key):
    return f"{prefix}labels_from_{key}.csv"


def predictions_are_up_to_date(predictions_path, model_folders, analyzer_path):
    """
    Saved predictions can be reused if they are newer than the models and than the analyzer's metrics.
    Retrained models are saved in new folders, so this mostly catches metrics which were recomputed.
    """
    predictions_path = Path(predictions_path)
    if not predictions_path.is_file():
        return False
    saved_time = predictions_path.stat().st_mtime

    source_files = [Path(model_folder) / "best_model.skops" for model_folder in model_folders]
    if not is_remote_path(analyzer_path):
        for extension_name in ["quality_metrics", "template_metrics"]:
            extension_folder = Path(analyzer_path) / "extensions" / extension_name
            if extension_folder.is_dir():
                source_files += [path for path in extension_folder.rglob("*") if path.is_file()]

    return all(saved_time > path.stat().st_mtime for path in source_files if path.is_file())


def is_remote_path(path):
    return "://" in str(path)


def predictions_from_table(table, sorting_analyzer, mode):
    """
    Rebuilds the outputs of `get_predictions` from a saved predictions table, so saved predictions can
    be reused instead of applying the models again. The human_label column is not a model output and is skipped.
    """
    # unit ids are read back from csv with pandas' inferred dtype, so match them as strings
    table_unit_ids = {str(unit_id): unit_id for unit_id in table.index}
    if set(table_unit_ids) != set(str(unit_id) for unit_id in sorting_analyzer.unit_ids):
        return None
    table = table.loc[[table_unit_ids[str(unit_id)] for unit_id in sorting_analyzer.unit_ids]]
    table.index = sorting_analyzer.unit_ids

    label_columns = {
        column: table[column].dropna().astype(str)
        for column in table.columns if column.endswith("_model_label")
    }
    confidence_columns = {column: table[column].values for column in table.columns if column.endswith("_confidence")}
    selection_confidence = selection_confidence_from_table(table, mode)
    return label_columns, confidence_columns, selection_confidence


def get_saved_or_new_predictions(sorting_analyzer, analyzer_path, mode, model_folders, predictions_path):
    """
    Returns the outputs of `get_predictions`, reusing the predictions saved at `predictions_path` when they
    are up to date, and otherwise applying the models and saving their predictions there.
    """
    predictions_path = Path(predictions_path)
    with PREDICTIONS_LOCK:
        if predictions_are_up_to_date(predictions_path, model_folders, analyzer_path):
            saved = predictions_from_table(pd.read_csv(predictions_path, index_col="unit_id"), sorting_analyzer, mode)
            if saved is not None:
                return saved

        label_columns, confidence_columns, selection_confidence = get_predictions(sorting_analyzer, mode, model_folders)
        # the folder is made with the analyzer (or by Apply). If it is gone, the analyzer was removed while the
        # predictions were being made (e.g. by the figures, in the background): don't make the folder again
        if predictions_path.parent.is_dir():
            save_csv_atomic(predictions_table(sorting_analyzer.unit_ids, label_columns, confidence_columns), predictions_path)
        return label_columns, confidence_columns, selection_confidence


def is_analyzer_folder(folder):
    """
    True for a SortingAnalyzer folder (binary or zarr), as spikeinterface_gui.main.check_folder_is_analyzer checks,
    which can't be used without the GUI installed.
    """
    folder = Path(folder)
    if not folder.is_dir():
        return False
    if folder.suffix == ".zarr":
        import zarr
        try:
            info = zarr.open(str(folder), mode="r").attrs.get("spikeinterface_info")
        except Exception:
            return False
    else:
        info_file = folder / "spikeinterface_info.json"
        if not info_file.is_file():
            return False
        try:
            info = json.loads(info_file.read_text())
        except (OSError, ValueError):
            return False
    return isinstance(info, dict) and info.get("object") == "SortingAnalyzer"


def find_analyzers(folder, max_depth=3):
    """Finds SortingAnalyzer folders inside `folder`, without looking inside the analyzers themselves."""

    found = []
    folder = Path(folder)
    if is_analyzer_folder(folder):
        return [folder]
    if max_depth == 0:
        return found
    try:
        subfolders = sorted(path for path in folder.iterdir() if path.is_dir() and not path.name.startswith("."))
    except PermissionError:
        return found
    for subfolder in subfolders:
        found += find_analyzers(subfolder, max_depth - 1)
    return found


def analyzer_output_names(analyzer_paths):
    """
    Short unique names for analyzers, used to name their output files. Analyzer folders often
    have the same name (e.g. sorting_analyzer), so parent folder names are added until they differ.
    Names longer than MAX_OUTPUT_NAME_LENGTH (e.g. of analyzers on s3) are shortened.
    """

    paths = [Path(str(analyzer_path).rstrip("/\\")) for analyzer_path in analyzer_paths]
    depth = 1
    while True:
        names = ["_".join(path.parts[-depth:]).replace(":", "").replace("\\", "").replace("/", "") for path in paths]
        if len(set(names)) == len(names) or depth >= max(len(path.parts) for path in paths):
            return [shorten_output_name(name) for name in names]
        depth += 1


# output file names also have the models' names in them: keep paths shorter than Windows allows (260 characters)
MAX_OUTPUT_NAME_LENGTH = 60


def shorten_output_name(name):
    """A name of at most MAX_OUTPUT_NAME_LENGTH characters: the start of a longer one, and a code which keeps it unique."""
    if len(name) <= MAX_OUTPUT_NAME_LENGTH:
        return name
    short_hash = hashlib.md5(name.encode()).hexdigest()[:8]
    return f"{name[:MAX_OUTPUT_NAME_LENGTH - 9]}_{short_hash}"


def apply_to_analyzers(analyzer_paths, mode, model_folders, key, output_folder, output_names=None,
                       compute_missing_metrics=False, job_kwargs=None):
    """
    Applies the model selection to each analyzer and saves its predictions to
    `{output_folder}/{analyzer name}__labels_from_{key}.csv`. Analyzers which can't be labelled,
    e.g. because a metric needed by a model is missing, are skipped and reported. With
    `compute_missing_metrics`, the metrics the models need are computed first and saved in the analyzer.

    The analyzers' file names start with `output_names` (default: short unique names from their paths).
    Returns a list of dicts with the analyzer `path`, its `output_name`, and the saved `file` (or the `error`).
    """
    import spikeinterface.full as si

    output_folder = Path(output_folder)
    output_folder.mkdir(parents=True, exist_ok=True)

    results = []
    if output_names is None:
        output_names = analyzer_output_names(analyzer_paths)
    for analyzer_path, output_name in zip(analyzer_paths, output_names):
        print(f"\nApplying models to {analyzer_path}...")
        result = dict(path=str(analyzer_path), output_name=output_name)
        output_file = output_folder / predictions_file_name(f"{output_name}__", key)
        try:
            sorting_analyzer = si.load_sorting_analyzer(analyzer_path, load_extensions=False)
            if compute_missing_metrics:
                # imported here, since metrics imports this module
                from unitrefine.metrics import compute_missing_metrics as compute_metrics
                compute_metrics(sorting_analyzer, model_folders, job_kwargs)
            # applying the same models to the same analyzer again reuses the saved predictions
            label_columns, confidence_columns, _ = get_saved_or_new_predictions(
                sorting_analyzer, analyzer_path, mode, model_folders, output_file
            )
        except Exception as e:
            result["error"] = describe_prediction_error(e)
            print(f"    Skipped: {result['error']}")
            results.append(result)
            continue

        table = predictions_table(sorting_analyzer.unit_ids, label_columns, confidence_columns)

        result["file"] = str(output_file)
        counts = table[list(label_columns.keys())[0]].value_counts().to_dict()
        print(f"    Predictions for {len(table)} units in {output_file}: {counts}")
        results.append(result)

    return results


def selection_confidence_from_table(table, mode):
    """The confidence of the whole selection for each unit, from a saved predictions table
    (see `get_predictions`): cascade confidence, or the lowest confidence over the binary models."""
    if mode == "cascade":
        return table["cascade_confidence"].values
    confidence_columns = [column for column in table.columns if column.endswith("_confidence")]
    return table[confidence_columns].min(axis=1).values


def load_or_compute_predictions(target, mode, model_folders, key):
    """
    Returns the saved predictions table of a relabel target (a dict with `path`, `save_folder` and
    `file_prefix`), computing and saving it first if Inspect or Apply has not done so yet.
    """
    import spikeinterface.full as si

    predictions_path = Path(target["save_folder"]) / predictions_file_name(target["file_prefix"], key)
    with PREDICTIONS_LOCK:
        if predictions_are_up_to_date(predictions_path, model_folders, target["path"]):
            return pd.read_csv(predictions_path, index_col="unit_id")

        print(f"No saved predictions for {target['path']}, applying the models...")
        sorting_analyzer = si.load_sorting_analyzer(target["path"], load_extensions=False)
        get_saved_or_new_predictions(sorting_analyzer, target["path"], mode, model_folders, predictions_path)
        return pd.read_csv(predictions_path, index_col="unit_id")


def relabelled_units_path(target, key):
    return Path(target["save_folder"]) / f"{target['file_prefix']}relabelled_units_{key}.csv"


def units_to_relabel_path(target, key):
    return Path(target["save_folder"]) / f"{target['file_prefix']}units_to_relabel_{key}.csv"


def find_least_confident_units(targets, mode, model_folders, key, num_units):
    """
    Ranks the units of all targets together by the confidence of the selection, and picks the
    `num_units` least confident ones which have no human label yet: units which were curated, imported,
    or relabelled in any earlier round (with any model) are never picked again, so each round asks about
    new units (sampling without replacement). For each target, the picked units are saved to
    `units_to_relabel_{key}.csv`, which Relabel opens.

    Returns {target index: number of picked units}, and a list of (target, error) for targets skipped.
    """

    candidates, skipped = [], []
    for target_index, target in enumerate(targets):
        try:
            table = load_or_compute_predictions(target, mode, model_folders, key)
        except Exception as e:
            skipped.append((target, describe_prediction_error(e)))
            continue

        # imported here, since evaluation imports this module
        from unitrefine.evaluation import human_labels
        already_labelled = set(human_labels(target))

        confidences = selection_confidence_from_table(table, mode)
        for unit_id, confidence in zip(table.index, confidences):
            if str(unit_id) not in already_labelled and not np.isnan(confidence):
                candidates.append((confidence, target_index, unit_id))

    candidates.sort(key=lambda candidate: candidate[0])
    picked = candidates[:num_units]

    num_picked = {}
    for target_index, target in enumerate(targets):
        target_units = [(unit_id, confidence) for confidence, index, unit_id in picked if index == target_index]
        num_picked[target_index] = len(target_units)
        if any(skipped_target is target for skipped_target, _ in skipped):
            continue
        save_csv_atomic(pd.DataFrame(target_units, columns=["unit_id", "confidence"]), units_to_relabel_path(target, key), index=False)

    return num_picked, skipped


def describe_prediction_error(error):
    """A readable message for errors raised while predicting, e.g. missing metrics."""
    if isinstance(error, OSError) and error.filename is not None:
        return f"{error.strerror}: {error.filename}"
    message = " ".join(str(arg) for arg in error.args) if len(error.args) > 0 else str(error)
    if "required metrics" in message:
        return f"{message}. Compute the missing metrics for this analyzer, or use a model which doesn't need them."
    return message
