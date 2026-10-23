"""
Data for the figures: compares models with the human labels they were not trained on, and computes
SHAP values. Kept free of Qt, so it can be used and tested on its own, and run in a background thread.
"""

import json
import os
from pathlib import Path

import numpy as np
import pandas as pd

from unitrefine.predictions import (
    describe_prediction_error,
    get_model_classes,
    get_saved_or_new_predictions,
    is_remote_path,
    metrics_for_model,
    as_model_input,
    predictions_are_up_to_date,
    map_quality_to_model_class,
    predictions_file_name,
)

TRAINED_ON_FILE = "trained_on.json"


# --------------------------------------------------------------------------------------------------
# which human labels a model was trained on
# --------------------------------------------------------------------------------------------------

def normalise_path(path):
    """Analyzer paths are compared in one form, so the same analyzer added in two ways still matches."""
    if is_remote_path(path):
        return str(path)
    return os.path.normcase(os.path.abspath(str(path)))


def write_trained_on(model_folder, units_by_analyzer, complete=True):
    """
    Saves which human-labelled units a model was trained on, as {analyzer path: [unit ids]}.
    `complete` is False when the model also learned from units which are not known (see `read_trained_on`).
    """
    data = {
        "complete": complete,
        "analyzers": {
            normalise_path(path): sorted(set(str(unit_id) for unit_id in unit_ids))
            for path, unit_ids in units_by_analyzer.items()
        },
    }
    with open(Path(model_folder) / TRAINED_ON_FILE, "w") as f:
        json.dump(data, f, indent=4)


def read_trained_on(model_folder):
    """
    Returns ({analyzer path: set of unit ids}, status), where status is
        "recorded":   trained in UnitRefine, the units are known
        "downloaded": a model from the Hugging Face Hub, not trained on any of the user's labels
        "unknown":    trained before UnitRefine recorded this; assumed trained on all curated labels
    """
    model_folder = Path(model_folder)
    trained_on_path = model_folder / TRAINED_ON_FILE
    if trained_on_path.is_file():
        with open(trained_on_path) as f:
            data = json.load(f)
        units = {path: set(unit_ids) for path, unit_ids in data["analyzers"].items()}
        return units, ("recorded" if data.get("complete", True) else "unknown")
    if (model_folder / "hfh_path.txt").is_file():
        return {}, "downloaded"
    return {}, "unknown"


def units_used_to_train(model_folder, targets):
    """
    The (analyzer path, unit id) pairs a model was trained on. For models with an unknown history,
    all curated (original) labels of the project's analyzers are assumed to have been used.
    """
    units, status = read_trained_on(model_folder)
    used = {(path, unit_id) for path, unit_ids in units.items() for unit_id in unit_ids}
    if status == "unknown":
        for target in targets:
            if target["analyzer_index"] is not None:
                for unit_id in original_labels(target):
                    used.add((normalise_path(target["path"]), unit_id))
    return used, status


# --------------------------------------------------------------------------------------------------
# human labels
# --------------------------------------------------------------------------------------------------

def original_labels(target):
    """
    The curated or imported labels of an analyzer, as {unit id: label}: for analyzers in the project (section 2),
    and labels imported for new analyzers in section 3.3, which models are never trained on.
    """
    labels_path = Path(target["save_folder"]) / f"{target['file_prefix']}labels.csv"
    if not labels_path.is_file():
        return {}
    labels = pd.read_csv(labels_path)
    return dict(zip(labels["unit_id"].astype(str), labels["quality"]))


def human_labels(target):
    """
    All human labels of an analyzer, as {unit id: label}: the curated or imported labels, overwritten by
    the units relabelled later (with any model selection; the most recent relabel wins).
    """
    labels = original_labels(target)
    relabel_files = Path(target["save_folder"]).glob(f"{target['file_prefix']}relabelled_units_*.csv")
    for relabel_file in sorted(relabel_files, key=lambda path: path.stat().st_mtime):
        relabels = pd.read_csv(relabel_file)
        labels.update(dict(zip(relabels["unit_id"].astype(str), relabels["quality"])))
    return {unit_id: label for unit_id, label in labels.items() if isinstance(label, str)}


# --------------------------------------------------------------------------------------------------
# comparing models with human labels
# --------------------------------------------------------------------------------------------------

# predictions already looked up, keyed by the analyzer, the models and the saved predictions file's
# modification time, so they are looked up again when the file changes
_predictions_cache = {}

# analyzers which could not be labelled while making a figure, as {path: reason}; cleared for each figure
skipped_analyzers = {}

# called with a short message about what is being computed, so a slow figure can say what it is doing
_progress_callback = None


def set_progress_callback(callback):
    global _progress_callback
    _progress_callback = callback


def report_progress(message):
    if _progress_callback is not None:
        _progress_callback(message)


def skipped_key(target, model_folders):
    """How a skipped analyzer is listed: its path and the model(s) which could not label it."""
    return f"{target['path']} (model: {', '.join(Path(model_folder).name for model_folder in model_folders)})"


def target_predictions(target, mode, model_folders, key):
    """
    Predictions of a model selection for one analyzer, reusing the saved ones when up to date.
    Returns the label and confidence columns (see `get_predictions`) and the analyzer's unit ids as strings,
    which the confidence arrays are aligned with.
    """
    import spikeinterface.full as si

    predictions_path = Path(target["save_folder"]) / predictions_file_name(target["file_prefix"], key)
    cache_key = (
        str(target["path"]), mode, tuple(str(model_folder) for model_folder in model_folders), key,
        file_modification_time(predictions_path),
    )
    if cache_key not in _predictions_cache:
        analyzer_name = Path(str(target["path"])).name
        if predictions_are_up_to_date(predictions_path, model_folders, target["path"]):
            report_progress(f"Reading the saved predictions of {analyzer_name}")
        else:
            model_names = ", ".join(Path(model_folder).name for model_folder in model_folders)
            report_progress(f"Applying {model_names} to {analyzer_name} (done once, then saved)")
        sorting_analyzer = si.load_sorting_analyzer(target["path"], load_extensions=False)
        label_columns, confidence_columns, _ = get_saved_or_new_predictions(
            sorting_analyzer, target["path"], mode, model_folders, predictions_path
        )
        # keyed by the file as it is now, which is new if the predictions were just made
        cache_key = cache_key[:-1] + (file_modification_time(predictions_path),)
        _predictions_cache[cache_key] = (label_columns, confidence_columns, [str(unit_id) for unit_id in sorting_analyzer.unit_ids])
    return _predictions_cache[cache_key]


def file_modification_time(path):
    try:
        return os.stat(path).st_mtime_ns
    except OSError:
        return None


def evaluate(targets, mode, model_folders, key, prediction_column, classes, excluded_units):
    """
    Compares the predictions in `prediction_column` with the human labels of all targets, leaving out
    the `excluded_units` (the ones the model was trained on). Human labels are mapped to the model's
    `classes` (e.g. MUA -> neural); labels the model doesn't use are skipped.

    Returns a DataFrame with one row per compared unit: analyzer, unit_id, human (in the model's
    classes) and predicted. Analyzers which can't be predicted (e.g. missing metrics) are skipped and
    recorded in `skipped_analyzers`.
    """
    rows = []
    for target in targets:
        labels = human_labels(target)
        analyzer_path = normalise_path(target["path"])
        labels = {unit_id: label for unit_id, label in labels.items() if (analyzer_path, unit_id) not in excluded_units}
        if len(labels) == 0:
            continue
        try:
            label_columns, _, _ = target_predictions(target, mode, model_folders, key)
        except Exception as e:
            skipped_analyzers[skipped_key(target, model_folders)] = describe_prediction_error(e)
            continue
        predictions = label_columns[prediction_column]
        predictions_by_unit = dict(zip(predictions.index.astype(str), predictions.values))
        for unit_id, label in labels.items():
            human_class = map_quality_to_model_class(label, classes)
            if human_class is None or unit_id not in predictions_by_unit:
                continue
            rows.append(dict(analyzer=target["path"], unit_id=unit_id, human=human_class, predicted=predictions_by_unit[unit_id]))
    return pd.DataFrame(rows, columns=["analyzer", "unit_id", "human", "predicted"])


def selection_evaluations(targets, mode, model_folders, key):
    """
    One evaluation per model of the selection, plus the combined cascade for a cascade. Each is a dict
    with `name`, `classes`, `comparison` (from `evaluate`), the trained-on `status`, the number of
    human-labelled units left out because the model was trained on them, and the number of human
    labels it could be compared with (`num_candidates`; if the comparison is empty but this is not,
    the model could not label the analyzers).
    """
    model_folders = [Path(model_folder) for model_folder in model_folders]
    evaluations = []

    def add(name, run_mode, run_model_folders, run_key, prediction_column, classes, excluded, status):
        comparison = evaluate(targets, run_mode, run_model_folders, run_key, prediction_column, classes, excluded)
        num_left_out, num_candidates = 0, 0
        for target in targets:
            analyzer_path = normalise_path(target["path"])
            for unit_id, label in human_labels(target).items():
                if (analyzer_path, unit_id) in excluded:
                    num_left_out += 1
                elif map_quality_to_model_class(label, classes) is not None:
                    num_candidates += 1
        evaluations.append(dict(
            name=name, classes=list(classes), comparison=comparison, status=status,
            num_left_out=num_left_out, num_candidates=num_candidates,
        ))

    trained = {model_folder: units_used_to_train(model_folder, targets) for model_folder in model_folders}

    if mode == "cascade":
        layer1, layer2 = model_folders
        excluded = trained[layer1][0] | trained[layer2][0]
        statuses = {trained[layer1][1], trained[layer2][1]}
        status = "unknown" if "unknown" in statuses else ("recorded" if "recorded" in statuses else "downloaded")
        add("cascade", mode, model_folders, key, "cascade_model_label", ["good", "MUA", "noise"], excluded, status)
        add(f"L1 {layer1.name}", mode, model_folders, key, f"L1_{layer1.name}_model_label", get_model_classes(layer1), *trained[layer1])
        add(f"L2 {layer2.name}", mode, model_folders, key, f"L2_{layer2.name}_model_label", get_model_classes(layer2), *trained[layer2])
    else:
        # each binary model is evaluated on its own, so a model which can't label an analyzer
        # (e.g. it needs metrics the analyzer doesn't have) doesn't stop the others
        for model_folder in model_folders:
            add(
                model_folder.name, "binary", [model_folder], model_folder.name, f"{model_folder.name}_model_label",
                get_model_classes(model_folder), *trained[model_folder],
            )

    return evaluations


def lineage_evaluations(targets, lineage_folders):
    """
    Evaluates each model of a retraining lineage (a model and its retrained versions, in order) on the
    same units: the human labels none of them were trained on, so the rounds can be compared.
    """
    excluded = set()
    statuses = []
    for model_folder in lineage_folders:
        used, status = units_used_to_train(model_folder, targets)
        excluded |= used
        statuses.append(status)

    results = []
    for model_folder in lineage_folders:
        model_folder = Path(model_folder)
        comparison = evaluate(
            targets, "binary", [model_folder], model_folder.name, f"{model_folder.name}_model_label",
            get_model_classes(model_folder), excluded,
        )
        results.append(dict(name=model_folder.name, comparison=comparison))
    return results, statuses


def scores(comparison):
    """Balanced accuracy, macro precision and macro recall of a comparison, or None if it is empty."""
    from sklearn.metrics import balanced_accuracy_score, precision_score, recall_score

    if len(comparison) == 0:
        return None
    y_true, y_pred = comparison["human"].values, comparison["predicted"].values
    return dict(
        balanced_accuracy=balanced_accuracy_score(y_true, y_pred),
        precision=precision_score(y_true, y_pred, average="macro", zero_division=0),
        recall=recall_score(y_true, y_pred, average="macro", zero_division=0),
    )


# --------------------------------------------------------------------------------------------------
# confidence
# --------------------------------------------------------------------------------------------------

def confidence_table(targets, mode, model_folders, key):
    """
    The predicted label and confidence of every unit of every analyzer, for each label column of the
    selection, with whether the unit was picked for relabelling. Returns {column: DataFrame}.
    """
    from unitrefine.predictions import units_to_relabel_path

    # a cascade is predicted as a whole; binary models one by one, so one failing model doesn't stop the others
    if mode == "cascade":
        runs = [(mode, model_folders, key)]
    else:
        runs = [("binary", [model_folder], Path(model_folder).name) for model_folder in model_folders]

    tables = {}
    for target in targets:
        # the units picked for relabelling belong to the whole selection
        picked = set()
        if units_to_relabel_path(target, key).is_file():
            picked = set(pd.read_csv(units_to_relabel_path(target, key))["unit_id"].astype(str))
        for run_mode, run_model_folders, run_key in runs:
            try:
                label_columns, confidence_columns, analyzer_unit_ids = target_predictions(target, run_mode, run_model_folders, run_key)
            except Exception as e:
                skipped_analyzers[skipped_key(target, run_model_folders)] = describe_prediction_error(e)
                continue
            # confidences are aligned with the analyzer's units; labels can cover fewer units (cascade layer 2)
            positions = {unit_id: position for position, unit_id in enumerate(analyzer_unit_ids)}
            for label_column, labels in label_columns.items():
                confidences = confidence_columns[label_column.replace("_model_label", "_confidence")]
                unit_ids = [str(unit_id) for unit_id in labels.index]
                rows = pd.DataFrame(dict(
                    analyzer=target["path"],
                    unit_id=unit_ids,
                    predicted=labels.values,
                    confidence=[confidences[positions[unit_id]] for unit_id in unit_ids],
                    picked=[unit_id in picked for unit_id in unit_ids],
                ))
                tables.setdefault(label_column, []).append(rows)
    return {column: pd.concat(rows, ignore_index=True) for column, rows in tables.items()}


# --------------------------------------------------------------------------------------------------
# SHAP
# --------------------------------------------------------------------------------------------------

def target_metrics(target):
    """The quality and template metrics of an analyzer, indexed by unit id (as strings)."""
    import spikeinterface.full as si
    from unitrefine.import_labels import compute_all_metrics

    metrics_path = Path(target["save_folder"]) / f"{target['file_prefix']}all_metrics.csv"
    if metrics_path.is_file():
        metrics = pd.read_csv(metrics_path, index_col="unit_id")
    else:
        metrics = compute_all_metrics(si.load_sorting_analyzer(target["path"], load_extensions=False))
    metrics.index = metrics.index.astype(str)
    return metrics


def recording_metrics(model_folder, targets):
    """
    The metrics a model uses, for all units of the analyzers which have them all, indexed by
    (analyzer path, unit id).
    """
    from spikeinterface.curation.model_based_curation import load_model

    pipeline, model_info = load_model(model_folder=model_folder, trust_model=True)
    metric_names = list(pipeline.feature_names_in_)

    metrics, paths = [], []
    for target in targets:
        report_progress(f"Reading the metrics of {Path(str(target['path'])).name}")
        try:
            # with the metric names of the SpikeInterface version the model was trained with
            target_metric_table = metrics_for_model(target_metrics(target), model_info)
        except Exception as e:
            skipped_analyzers[skipped_key(target, [model_folder])] = str(e)
            continue
        missing_metrics = [metric_name for metric_name in metric_names if metric_name not in target_metric_table.columns]
        if len(missing_metrics) > 0:
            skipped_analyzers[skipped_key(target, [model_folder])] = f"missing metrics used by the model: {missing_metrics}"
            continue
        metrics.append(as_model_input(target_metric_table[metric_names]))
        paths.append(str(target["path"]))
    if len(metrics) == 0:
        raise ValueError("None of the analyzers have all the metrics this model uses.")
    return pd.concat(metrics, keys=paths)


def training_metrics(model_folder):
    """The metrics of the units a model was trained on (its training_data.csv), in the model's metric order."""
    from spikeinterface.curation.model_based_curation import load_model

    pipeline, _ = load_model(model_folder=model_folder, trust_model=True)
    training_data = pd.read_csv(Path(model_folder) / "training_data.csv")
    return as_model_input(training_data[list(pipeline.feature_names_in_)])


def preferred_class(classes):
    """The class SHAP plots explain by default: the one most people care about."""
    for preferred in ["good", "sua", "SUA", "neural"]:
        if preferred in classes:
            return preferred
    return list(classes)[0]


def shap_explanation(model_folder, metrics, class_name, max_units=300, seed=0):
    """
    SHAP values of a model for one of its classes, on (up to `max_units`) units with the given `metrics`
    (a table with the model's metrics as columns). Returns a shap.Explanation whose data are the metric
    values after imputation and before scaling, so plots show real metric values.
    """
    import shap
    from spikeinterface.curation.model_based_curation import load_model

    pipeline, model_info = load_model(model_folder=model_folder, trust_model=True)
    metric_names = list(pipeline.feature_names_in_)
    metrics = metrics[metric_names]
    if len(metrics) > max_units:
        metrics = metrics.sample(max_units, random_state=seed)

    report_progress(f"Computing SHAP values of {Path(model_folder).name} on {len(metrics)} units")
    imputed = pipeline["imputer"].transform(metrics)
    scaled = pipeline["scaler"].transform(imputed)
    classifier = pipeline["classifier"]

    try:
        values = shap.TreeExplainer(classifier)(scaled, check_additivity=False).values
    except Exception:
        try:
            values = shap.LinearExplainer(classifier, scaled)(scaled).values
        except Exception:
            # other models: explain the predicted probabilities with a small background, on fewer units
            background = shap.kmeans(scaled, min(20, len(scaled)))
            explainer = shap.KernelExplainer(classifier.predict_proba, background)
            raw_values = explainer.shap_values(scaled[:100], nsamples=100, silent=True)
            imputed = imputed[:100]
            values = np.stack(raw_values, axis=-1) if isinstance(raw_values, list) else np.asarray(raw_values)

    # the classifier's outputs follow its classes_, which are the label codes of label_conversion
    label_conversion = model_info["label_conversion"]
    class_code = [int(code) for code, name in label_conversion.items() if name == class_name][0]
    class_position = [int(code) for code in classifier.classes_].index(class_code)
    if values.ndim == 3:
        class_values = values[:, :, class_position]
    elif class_position == 0:
        # single output models explain the second class; flip the sign for the first
        class_values = -values
    else:
        class_values = values

    return shap.Explanation(values=class_values, data=np.asarray(imputed), feature_names=metric_names)
