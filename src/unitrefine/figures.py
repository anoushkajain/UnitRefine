"""
The figures shown in sections 2, 3 and 4 of the main window.

Each figure is described by a `FigureSpec`: how to make it, where to save it, and which files it is made
from. A saved figure is reused while those files don't change, so the window opens quickly; otherwise it
is made again in a background thread. Figures are drawn with matplotlib's object oriented API (no pyplot),
so they can be made outside the main thread. Kept free of Qt.
"""

import hashlib
import json
import os
import threading
import time
import traceback
from dataclasses import dataclass
from pathlib import Path

import matplotlib
import numpy as np
import pandas as pd
from matplotlib import font_manager
from matplotlib.figure import Figure

from unitrefine import evaluation, unwritable_file_message
from unitrefine.predictions import (
    get_model_classes,
    map_quality_to_model_class,
    predictions_file_name,
    relabelled_units_path,
    selection_confidence_from_table,
    selection_key,
    units_to_relabel_path,
    load_or_compute_predictions,
)

# saved figures made with an older version of this file are made again
FIGURES_VERSION = 8

FIGURES_FOLDER = "figures"
TEST_PREDICTIONS_FILE = "test_predictions.csv"

# the same colour for a class in every figure
CLASS_COLOURS = {
    "good": "#12B76A", "sua": "#12B76A", "SUA": "#12B76A",
    "MUA": "#F79009", "mua": "#F79009",
    "noise": "#98A2B3",
    "neural": "#2E90FA",
    "not-sua": "#9E77ED",
}
CLASS_ORDER = ["good", "sua", "SUA", "neural", "MUA", "mua", "not-sua", "noise"]
HUMAN_LABELS = ["good", "MUA", "noise"]

STATUS_TEXT = {
    "downloaded": "a downloaded model, never trained on your labels, so it is compared with all your labels",
    "recorded": "trained in UnitRefine; the labels it was trained on are left out",
    "unknown": "trained before UnitRefine recorded what models learn from, so all your curated labels are "
               "assumed to have been used and are left out",
}


def use_app_font():
    """Figures use the GUI's font (Inter, in resources/fonts) and a light, modern style."""
    fonts_folder = Path(__file__).absolute().parent / "resources" / "fonts"
    for font_file in fonts_folder.glob("*.ttf"):
        try:
            font_manager.fontManager.addfont(str(font_file))
        except Exception:
            pass
    families = {font.name for font in font_manager.fontManager.ttflist}
    matplotlib.rcParams.update({
        "font.family": ["Inter" if "Inter" in families else "DejaVu Sans", "DejaVu Sans"],
        "font.size": 9,
        "axes.titlesize": 10,
        "axes.titleweight": "bold",
        "axes.titlelocation": "left",
        "axes.labelsize": 9,
        "axes.edgecolor": "#98A2B3",
        "axes.labelcolor": "#344054",
        "axes.spines.top": False,
        "axes.spines.right": False,
        "text.color": "#1D2939",
        "xtick.color": "#475467",
        "ytick.color": "#475467",
        "legend.frameon": False,
        "figure.facecolor": "white",
        "savefig.facecolor": "white",
    })


use_app_font()


class NoData(Exception):
    """The figure can't be made yet; the message says what to do first. Shown as a hint, not an error."""


@dataclass
class FigureSpec:
    key: str             # unique within a section
    title: str
    path: Path           # where the png is saved
    make: object         # make() -> (Figure, [notes]); raises NoData
    inputs: object       # inputs() -> the files the figure is made from
    caption: str = ""    # how to read the figure
    params: str = ""     # anything else it depends on
    slow: bool = False   # takes a few seconds (e.g. SHAP)
    analyzers: tuple = ()  # the analyzers in the order they are listed in the GUI, which numbers them


# --------------------------------------------------------------------------------------------------
# saving, and reusing saved figures
# --------------------------------------------------------------------------------------------------

def file_state(path):
    try:
        stat = os.stat(path)
        return [str(path), stat.st_mtime_ns, stat.st_size]
    except OSError:
        return [str(path), None]


def signature(spec):
    """Changes whenever a file the figure is made from changes, or the figure's code changes."""
    state = [FIGURES_VERSION, spec.params, [file_state(path) for path in spec.inputs()]]
    return hashlib.md5(json.dumps(state, default=str).encode()).hexdigest()


def sidecar_path(spec):
    return spec.path.with_suffix(".json")


def cached_result(spec):
    """The saved result of a figure if it is still up to date, else None."""
    try:
        with open(sidecar_path(spec)) as f:
            result = json.load(f)
    except (OSError, ValueError):
        return None
    if result.get("signature") != signature(spec):
        return None
    # errors may be passing (e.g. a file being written), so a figure which failed is always made again
    if result.get("status") == "error":
        return None
    if result.get("status") == "done" and not spec.path.is_file():
        return None
    return result


# --------------------------------------------------------------------------------------------------
# where a figure's data comes from: printed at the bottom of every figure
# --------------------------------------------------------------------------------------------------

# the analyzers and models the figure being made uses; reset for each figure (figures are made one at a time)
_provenance = dict(data={}, models=[], numbering=[])

# above this many analyzers, dots are numbered (the numbers are listed at the bottom) instead of named
MAX_NAMED_DOTS = 8


def reset_provenance(analyzers=()):
    """`analyzers` number the analyzers (1, 2, ...), so an analyzer has the same number in every figure."""
    _provenance["data"] = {}
    _provenance["models"] = []
    _provenance["numbering"] = [evaluation.normalise_path(path) for path in analyzers]


def target_group(target):
    """Analyzers in the project (section 2) are used for training; the ones in 3.3 are new."""
    return "training" if target["analyzer_index"] is not None else "new"


def note_data(path, group, num_units):
    """
    Records that the figure uses `num_units` units of the analyzer at `path` ("training" or "new"), and
    returns the analyzer's number, which the bottom of the figure lists and dots can be labelled with.
    """
    data = _provenance["data"]
    key = evaluation.normalise_path(path)
    if key not in data:
        numbering = _provenance["numbering"]
        if key in numbering:
            number = numbering.index(key) + 1
        else:
            # e.g. an analyzer a model was trained on which is no longer in the project
            number = max([len(numbering)] + [info["number"] for info in data.values()]) + 1
        data[key] = dict(path=str(path), group=group, units=0, number=number)
    data[key]["units"] = max(data[key]["units"], int(num_units))
    return data[key]["number"]


def data_number(path):
    return _provenance["data"][evaluation.normalise_path(path)]["number"]


def note_models(*names):
    for name in names:
        if name not in _provenance["models"]:
            _provenance["models"].append(name)


def note_selection(mode, model_folders):
    names = [Path(model_folder).name for model_folder in model_folders]
    if mode == "cascade":
        note_models(f"cascade of {names[0]} (layer 1) and {names[1]} (layer 2)")
    else:
        note_models(*names)


def add_footer(figure):
    """Writes at the bottom of the figure which analyzers (numbered) and models it comes from."""
    import textwrap

    data, models = _provenance["data"], _provenance["models"]
    parts = []
    for group, title in [("training", "Analyzers used for training"), ("new", "New analyzers")]:
        items = [f"[{info['number']}] {analyzer_name(info['path'])} ({info['units']} units)"
                 for info in sorted(data.values(), key=lambda info: info["number"]) if info["group"] == group]
        if len(items) > 0:
            parts.append(f"{title}: " + ", ".join(items))
    if len(models) > 0:
        parts.append(("Model: " if len(models) == 1 else "Models: ") + ", ".join(models))
    if len(parts) == 0:
        return
    width = max(60, int(figure.get_figwidth() * 15))
    text = "\n".join(textwrap.fill(part, width) for part in parts)
    # a figure-level label, so the figure's layout makes room for it
    figure.supxlabel(text, x=0.005, ha="left", fontsize=7.5, color="#475467", fontweight="normal", linespacing=1.5)


def label_dot(ax, x, y, path, num_analyzers, colour="#1D2939", horizontal=False):
    """Labels an analyzer's dot: its name when there are few analyzers, else its number (listed at the bottom)."""
    number = data_number(path)
    if num_analyzers <= MAX_NAMED_DOTS:
        ax.annotate(f"[{number}] {short(analyzer_name(path), 22)}", (x, y), textcoords="offset points", xytext=(7, 0),
                    va="center", fontsize=6.5, color=colour, zorder=5,
                    bbox=dict(boxstyle="round,pad=0.15", facecolor="white", edgecolor="none", alpha=0.85))
    else:
        ax.annotate(str(number), (x, y), textcoords="offset points", xytext=(5, 3), fontsize=6.5, color=colour, zorder=5,
                    bbox=dict(boxstyle="round,pad=0.1", facecolor="white", edgecolor="none", alpha=0.85))


def skipped_notes():
    return [f"Left out {path}: {reason}" for path, reason in evaluation.skipped_analyzers.items()]


# a figure records what it uses in module-level state (_provenance, evaluation.skipped_analyzers), which each
# figure starts by resetting: figures are made one at a time, also when several threads make them (e.g. the
# figures thread of a window, and a test)
RENDER_LOCK = threading.RLock()


def render(spec, progress=None):
    """
    Makes and saves a figure. Returns a dict with the `status` (done, nodata or error), `notes` about
    the data (e.g. analyzers left out), a `message` for nodata and error, and the `seconds` it took.
    """
    with RENDER_LOCK:
        return render_unlocked(spec, progress)


def render_unlocked(spec, progress=None):
    start_time = time.time()
    evaluation.skipped_analyzers.clear()
    reset_provenance(spec.analyzers)
    evaluation.set_progress_callback(progress)
    try:
        spec.path.parent.mkdir(parents=True, exist_ok=True)
        figure, notes = spec.make()
        add_footer(figure)
        temporary_path = spec.path.with_name(spec.path.stem + ".tmp.png")
        # high enough resolution to stay sharp when zoomed in the viewer
        figure.savefig(temporary_path, dpi=200)
        os.replace(temporary_path, spec.path)
        result = dict(status="done", notes=list(notes) + skipped_notes())
    except NoData as e:
        # don't leave an out of date figure behind
        try:
            spec.path.unlink(missing_ok=True)
        except OSError:
            pass
        result = dict(status="nodata", message=str(e), notes=skipped_notes())
    except PermissionError as e:
        # e.g. a read-only file in the figures folder: say how to fix it
        result = dict(status="error", message=unwritable_file_message(e.filename or spec.path, e), notes=skipped_notes())
    except Exception as e:
        traceback.print_exc()
        result = dict(status="error", message=f"{type(e).__name__}: {e}", notes=skipped_notes())
    finally:
        evaluation.set_progress_callback(None)
    result["seconds"] = round(time.time() - start_time, 1)
    result["signature"] = signature(spec)
    result["title"] = spec.title
    try:
        with open(sidecar_path(spec), "w") as f:
            json.dump(result, f, indent=4)
    except OSError as e:
        # the figure is still shown; it is made again next time, as its record could not be saved
        print(unwritable_file_message(sidecar_path(spec), e))
    return result


# --------------------------------------------------------------------------------------------------
# drawing helpers
# --------------------------------------------------------------------------------------------------

def new_figure(width, height):
    return Figure(figsize=(width, height), layout="constrained")


def ordered_classes(classes):
    classes = list(dict.fromkeys(classes))
    return sorted(classes, key=lambda name: (CLASS_ORDER.index(name) if name in CLASS_ORDER else len(CLASS_ORDER), str(name)))


def class_colour(class_name, index=0):
    return CLASS_COLOURS.get(class_name, matplotlib.colormaps["tab10"](index % 10))


def short(text, length=38):
    text = str(text)
    return text if len(text) <= length else text[: length - 1] + "…"


def short_file_name(name, length=32):
    """File and folder names short enough to keep figure paths under Windows' 260 character limit."""
    name = str(name)
    if len(name) <= length:
        return name
    return f"{name[: length - 9]}_{hashlib.md5(name.encode()).hexdigest()[:8]}"


def analyzer_name(path):
    """Analyzer folders are often all called e.g. sorting_analyzer, so their parent folder is shown too."""
    parts = Path(str(path).rstrip("/\\")).parts
    return short("/".join(parts[-2:]), 34)


def message_axes(ax, text):
    ax.text(0.5, 0.5, text, ha="center", va="center", color="#667085", wrap=True)
    ax.axis("off")


def draw_confusion(ax, true_labels, predicted_labels, row_classes, column_classes, title, xlabel, ylabel):
    """A confusion matrix with the number of units and the share of its row in each cell."""
    counts = np.zeros((len(row_classes), len(column_classes)), dtype=int)
    for true_label, predicted_label in zip(true_labels, predicted_labels):
        if true_label in row_classes and predicted_label in column_classes:
            counts[row_classes.index(true_label), column_classes.index(predicted_label)] += 1
    row_totals = counts.sum(axis=1, keepdims=True)
    fractions = np.divide(counts, row_totals, out=np.zeros(counts.shape), where=row_totals > 0)
    ax.imshow(fractions, cmap="Blues", vmin=0, vmax=1, aspect="auto")
    for row in range(len(row_classes)):
        for column in range(len(column_classes)):
            colour = "white" if fractions[row, column] > 0.6 else "#1D2939"
            text = f"{counts[row, column]}\n{fractions[row, column]:.0%}" if row_totals[row, 0] > 0 else "0"
            ax.text(column, row, text, ha="center", va="center", color=colour, fontsize=9)
    ax.set_xticks(range(len(column_classes)), column_classes)
    ax.set_yticks(range(len(row_classes)), [f"{name} ({row_totals[index, 0]})" for index, name in enumerate(row_classes)])
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.set_title(title, fontsize=9.5)
    ax.spines[:].set_visible(False)
    ax.tick_params(length=0)
    return counts


def balanced_accuracy(true_labels, predicted_labels):
    from sklearn.metrics import balanced_accuracy_score
    return balanced_accuracy_score(list(true_labels), list(predicted_labels))


def bootstrap_balanced_accuracy(true_labels, predicted_labels, num_resamples=500, seed=0):
    """
    Balanced accuracy on `num_resamples` resamples (with replacement) of the units: how much the score could
    vary with another set of units of the same kind. Computed like sklearn's balanced_accuracy_score: the
    mean, over the labels present, of the share of units of that label the model got right.
    """
    true_labels = np.asarray(list(true_labels), dtype=object)
    predicted_labels = np.asarray(list(predicted_labels), dtype=object)
    rng = np.random.default_rng(seed)
    scores = []
    for _ in range(num_resamples):
        indices = rng.integers(0, len(true_labels), len(true_labels))
        resampled_true, resampled_predicted = true_labels[indices], predicted_labels[indices]
        recalls = [np.mean(resampled_predicted[resampled_true == label] == label) for label in np.unique(resampled_true)]
        scores.append(np.mean(recalls))
    return np.array(scores)


def draw_boxes(ax, groups, positions, colours, horizontal=False, width=0.55, dots=True, seed=0):
    """Box plots (median, quartiles, whiskers to 1.5 IQR) with each value as a dot, as in the UnitRefine paper."""
    groups = [np.asarray(group, dtype=float) for group in groups]
    keep = [index for index, group in enumerate(groups) if len(group) > 0]
    if len(keep) == 0:
        return
    boxes = ax.boxplot(
        [groups[index] for index in keep], positions=[positions[index] for index in keep], widths=width,
        orientation="horizontal" if horizontal else "vertical", patch_artist=True, showfliers=False,
        medianprops=dict(color="#1D2939", linewidth=1.4), whiskerprops=dict(color="#667085"),
        capprops=dict(color="#667085"), boxprops=dict(linewidth=1),
    )
    for patch, index in zip(boxes["boxes"], keep):
        patch.set_facecolor(matplotlib.colors.to_rgba(colours[index], 0.35))
        patch.set_edgecolor(colours[index])
    if dots:
        rng = np.random.default_rng(seed)
        for index in keep:
            jitter = rng.uniform(-width * 0.3, width * 0.3, len(groups[index]))
            if horizontal:
                ax.scatter(groups[index], positions[index] + jitter, s=7, color=colours[index], alpha=0.6, linewidths=0, zorder=3)
            else:
                ax.scatter(positions[index] + jitter, groups[index], s=7, color=colours[index], alpha=0.6, linewidths=0, zorder=3)


# metric groups and colours of the UnitRefine paper's AUC-ROC figure
PAPER_QUALITY_METRICS = [
    'firing_rate', 'presence_ratio', 'snr', 'isi_violations_ratio', 'isi_violations_count', 'rp_contamination',
    'rp_violations', 'sliding_rp_violation', 'amplitude_cutoff', 'drift_ptp', 'isolation_distance', 'l_ratio',
    'd_prime', 'sd_ratio', 'silhouette', 'nn_hit_rate', 'nn_miss_rate', 'amplitude_median', 'num_spikes',
    'drift_std', 'drift_mad',
]
PAPER_NEW_METRICS = ['amplitude_cv_median', 'amplitude_cv_range', 'sync_spike_2', 'sync_spike_4', 'sync_spike_8', 'firing_range']
METRIC_GROUP_COLOURS = {"quality": "#E74C3C", "new": "#3498DB", "template": "#2ECC71"}
METRIC_GROUP_NAMES = {"quality": "Quality metrics", "new": "New metrics", "template": "Template metrics"}


def metric_group(metric_name):
    """Quality, new (introduced with UnitRefine) or template metric, as grouped in the paper."""
    if metric_name in PAPER_QUALITY_METRICS:
        return "quality"
    if metric_name in PAPER_NEW_METRICS:
        return "new"
    return "template"


def draw_shap(figure, explanation, class_name, max_display=12):
    """Mean |SHAP| per metric (left) and a beeswarm of every unit (right), for the top metrics."""
    values = np.asarray(explanation.values)
    data = np.asarray(explanation.data, dtype=float)
    names = list(explanation.feature_names)
    mean_impact = np.abs(values).mean(axis=0)
    order = np.argsort(mean_impact)[::-1][:max_display]

    bar_ax, swarm_ax = figure.subplots(1, 2, width_ratios=[1, 1.5], sharey=True)
    y_positions = np.arange(len(order))[::-1]
    bar_ax.barh(y_positions, mean_impact[order], color="#6172F3", height=0.65)
    bar_ax.set_yticks(y_positions, [short(names[index], 28) for index in order])
    bar_ax.set_xlabel("Average impact (mean |SHAP value|)")
    bar_ax.set_title("Most important metrics")
    bar_ax.tick_params(axis="y", length=0)

    rng = np.random.default_rng(0)
    colour_map = matplotlib.colormaps["coolwarm"]
    for y, index in zip(y_positions, order):
        shap_values = values[:, index]
        metric_values = data[:, index]
        # colour by the rank of the metric value, so a few extreme values don't wash out the colours
        ranks = pd.Series(metric_values).rank(pct=True).values
        # spread the dots vertically where they are dense, like a beeswarm
        bins = np.histogram_bin_edges(shap_values, bins=30)
        bin_index = np.clip(np.digitize(shap_values, bins) - 1, 0, len(bins) - 2)
        density = np.bincount(bin_index, minlength=len(bins) - 1)[bin_index].astype(float)
        spread = 0.38 * density / max(density.max(), 1)
        jitter = rng.uniform(-1, 1, len(shap_values)) * spread
        colours = [colour_map(rank) if not np.isnan(rank) else (0.75, 0.75, 0.75, 1) for rank in ranks]
        swarm_ax.scatter(shap_values, y + jitter, c=colours, s=7, linewidths=0, alpha=0.85, rasterized=True)
    swarm_ax.axvline(0, color="#98A2B3", linewidth=0.8)
    swarm_ax.set_xlabel(f"SHAP value (→ more likely '{class_name}')")
    swarm_ax.set_title(f"How metric values push units towards '{class_name}'")
    swarm_ax.tick_params(axis="y", length=0)
    colour_bar = figure.colorbar(
        matplotlib.cm.ScalarMappable(cmap=colour_map, norm=matplotlib.colors.Normalize(0, 1)),
        ax=swarm_ax, ticks=[0, 1], aspect=30, pad=0.01,
    )
    colour_bar.ax.set_yticklabels(["low", "high"])
    colour_bar.set_label("Metric value", labelpad=-8)
    colour_bar.outline.set_visible(False)


# --------------------------------------------------------------------------------------------------
# section 2: your labels and the models trained on them
# --------------------------------------------------------------------------------------------------

def project_label_files(targets):
    files = []
    for target in targets:
        folder = Path(target["save_folder"])
        files += [folder / "labels.csv", folder / "labelled_metrics.csv", folder / "all_metrics.csv"]
    return files


def read_labels(target):
    labels_path = Path(target["save_folder"]) / "labels.csv"
    if not labels_path.is_file():
        return None
    labels = pd.read_csv(labels_path)
    labels["unit_id"] = labels["unit_id"].astype(str)
    return labels.dropna(subset=["quality"])


def labelled_metrics_table(targets):
    """The metrics and label of every curated or imported unit, with an `analyzer` and a `label` column."""
    tables = []
    for target in targets:
        labels = read_labels(target)
        metrics_path = Path(target["save_folder"]) / "labelled_metrics.csv"
        if labels is None or not metrics_path.is_file():
            continue
        metrics = pd.read_csv(metrics_path, index_col=0)
        metrics.index = metrics.index.astype(str)
        quality = dict(zip(labels["unit_id"], labels["quality"]))
        metrics = metrics[metrics.index.isin(list(quality))]
        metrics = metrics.drop(columns=[column for column in ["quality", "unit_id", "index"] if column in metrics.columns])
        metrics = metrics.select_dtypes(include="number")
        metrics.insert(0, "label", [quality[unit_id] for unit_id in metrics.index])
        metrics.insert(0, "analyzer", str(target["path"]))
        note_data(target["path"], target_group(target), len(metrics))
        tables.append(metrics)
    if len(tables) == 0:
        return None
    return pd.concat(tables)


# dots of analyzers used for training and of new analyzers have different markers
ANALYZER_GROUP_MARKERS = {"training": ("o", "analyzers used for training"), "new": ("^", "new analyzers")}


def draw_label_shares(ax, rows, classes):
    """
    Box plots of the share of units with each label, over the analyzers: one dot per analyzer, so the plot
    builds up as analyzers are added. `rows` are (analyzer path, {label: count}, group), group being
    "training" or "new" (see ANALYZER_GROUP_MARKERS). Each dot is labelled with its analyzer.
    """
    colours = [class_colour(name, index) for index, name in enumerate(classes)]
    shares = [[100 * counts.get(class_name, 0) / max(sum(counts.values()), 1) for _, counts, _ in rows] for class_name in classes]
    draw_boxes(ax, shares, list(range(len(classes))), colours, width=0.55, dots=False)
    rng = np.random.default_rng(0)
    groups_present = []
    for class_index, class_name in enumerate(classes):
        for (path, _, group), share in zip(rows, shares[class_index]):
            marker = ANALYZER_GROUP_MARKERS[group][0]
            x = class_index + rng.uniform(-0.12, 0.12)
            ax.scatter(x, share, marker=marker, s=26, color=colours[class_index], edgecolors="#1D2939", linewidths=0.5, zorder=3)
            label_dot(ax, x, share, path, len(rows))
            if group not in groups_present:
                groups_present.append(group)
    totals = [sum(counts.get(class_name, 0) for _, counts, _ in rows) for class_name in classes]
    ax.set_xticks(range(len(classes)), [f"{name}\n({total} units)" for name, total in zip(classes, totals)])
    ax.set_xlim(-0.6, len(classes) - 0.1)
    ax.set_ylabel("% of an analyzer's units")
    ax.set_ylim(-3, 103)
    ax.grid(axis="y", alpha=0.25)
    if len(groups_present) > 1:
        handles = [matplotlib.lines.Line2D([0], [0], marker=ANALYZER_GROUP_MARKERS[group][0], color="white",
                                           markerfacecolor="#98A2B3", markeredgecolor="#1D2939", markersize=7,
                                           label=ANALYZER_GROUP_MARKERS[group][1]) for group in groups_present]
        ax.legend(handles=handles, loc="upper right", fontsize=8)


def make_label_counts(targets):
    rows = []
    for target in targets:
        labels = read_labels(target)
        if labels is not None and len(labels) > 0:
            note_data(target["path"], "training", len(labels))
            rows.append((target["path"], labels["quality"].value_counts().to_dict(), "training"))
    if len(rows) == 0:
        raise NoData("Curate or import labels to see how many units of each label you have.")
    classes = ordered_classes([class_name for _, counts, _ in rows for class_name in counts])
    num_units = sum(sum(counts.values()) for _, counts, _ in rows)

    figure = new_figure(6.5, 4.4)
    ax = figure.add_subplot()
    draw_label_shares(ax, rows, classes)
    ax.set_title(f"Your labels: {num_units} units in {len(rows)} analyzer{'s' if len(rows) > 1 else ''}")
    return figure, []


def kruskal_effect_sizes(table, metric_names):
    """How much each metric differs between the labels: the Kruskal–Wallis H test's effect size (ε²)."""
    from scipy.stats import kruskal

    effect_sizes = {}
    for metric_name in metric_names:
        groups = [group[metric_name].dropna().values for _, group in table.groupby("label")]
        groups = [group for group in groups if len(group) >= 3]
        num_values = sum(len(group) for group in groups)
        if len(groups) < 2 or len(np.unique(np.concatenate(groups))) < 2:
            continue
        statistic, _ = kruskal(*groups)
        effect_sizes[metric_name] = (statistic - len(groups) + 1) / (num_values - len(groups))
    return effect_sizes


def make_metric_distributions(targets, num_metrics=9):
    table = labelled_metrics_table(targets)
    if table is None or len(table) < 6:
        raise NoData("Curate or import labels to see how the metrics differ between your labels.")
    metric_names = [column for column in table.columns if column not in ("analyzer", "label")]
    effect_sizes = kruskal_effect_sizes(table, metric_names)
    if len(effect_sizes) == 0:
        raise NoData("Label units of at least two different labels (at least 3 units each) to compare the metrics.")
    # metrics which don't differ between the labels at all (ε² <= 0) are left out
    ranked = [name for name in sorted(effect_sizes, key=effect_sizes.get, reverse=True) if effect_sizes[name] > 0][:num_metrics]
    if len(ranked) == 0:
        raise NoData("None of the metrics differ between your labels.")
    classes = ordered_classes(table["label"].unique())
    colours = [class_colour(name, index) for index, name in enumerate(classes)]

    num_columns = 3
    num_rows = int(np.ceil(len(ranked) / num_columns))
    figure = new_figure(9, 0.5 + 2.3 * num_rows)
    axes = figure.subplots(num_rows, num_columns, squeeze=False).ravel()
    for ax, metric_name in zip(axes, ranked):
        values = table[metric_name].replace([np.inf, -np.inf], np.nan)
        groups = [values[table["label"] == class_name].dropna().values for class_name in classes]
        finite = values.dropna()
        low, high = np.percentile(finite, [1, 99])
        # metrics spanning orders of magnitude (e.g. firing rate) are shown on a log scale
        use_log = low > 0 and high / low > 100
        if use_log:
            ax.set_yscale("log")
            groups = [group[group > 0] for group in groups]
        draw_boxes(ax, groups, list(range(len(classes))), colours)
        if high > low:
            # a few extreme units would squash the boxes, so the axis shows the 1st-99th percentile
            padding = (high / low) ** 0.05 if use_log else (high - low) * 0.05
            ax.set_ylim((low / padding, high * padding) if use_log else (low - padding, high + padding))
        ax.set_xticks(range(len(classes)), classes)
        ax.set_title(f"{short(metric_name, 30)}   ε² = {effect_sizes[metric_name]:.2f}", fontsize=9)
        ax.grid(axis="y", alpha=0.25)
    for ax in axes[len(ranked):]:
        ax.axis("off")
    counts = table["label"].value_counts()
    handles = [matplotlib.patches.Patch(color=colour, alpha=0.6) for colour in colours]
    figure.legend(handles, [f"{name} ({counts.get(name, 0)})" for name in classes], loc="outside upper right", ncol=len(classes))
    figure.suptitle("Metrics that differ most between your labels", x=0.01, ha="left", fontweight="bold", fontsize=10)
    return figure, []


def roc_auc_from_ranks(is_positive, values):
    """The AUC-ROC from ranks (Mann–Whitney U): the same as sklearn's roc_auc_score, ties included, and much faster."""
    from scipy.stats import rankdata

    is_positive = np.asarray(is_positive, dtype=bool)
    ranks = rankdata(values)
    num_positive = is_positive.sum()
    num_negative = len(is_positive) - num_positive
    return (ranks[is_positive].sum() - num_positive * (num_positive + 1) / 2) / (num_positive * num_negative)


def metric_auc_values(table, positive_label, metric_names, min_units=3, num_resamples=200, seed=0):
    """
    The AUC-ROC of each metric on its own for telling `positive_label` units from the others, as in the
    UnitRefine paper: one value per analyzer with at least `min_units` units of each kind. With fewer than 3
    such analyzers there are too few values for a box plot, so the units of all analyzers are resampled
    (`num_resamples` times, with replacement) instead. Returns ({metric: values},
    {metric: the analyzer of each value, None for resamples}, method).
    """
    # plain arrays, so the many resamples are quick
    all_values = table[metric_names].replace([np.inf, -np.inf], np.nan).values.astype(float)
    all_positive = (table["label"] == positive_label).values
    analyzers = table["analyzer"].values

    def aucs(indices):
        is_positive = all_positive[indices]
        values = {}
        for column, metric_name in enumerate(metric_names):
            metric_values = all_values[indices, column]
            finite = ~np.isnan(metric_values)
            positives = is_positive[finite]
            if positives.sum() >= min_units and (~positives).sum() >= min_units and np.ptp(metric_values[finite]) > 0:
                values[metric_name] = roc_auc_from_ranks(positives, metric_values[finite])
        return values

    per_analyzer = [(analyzer, aucs(np.flatnonzero(analyzers == analyzer))) for analyzer in pd.unique(analyzers)]
    per_analyzer = [(analyzer, values) for analyzer, values in per_analyzer if len(values) > 0]
    if len(per_analyzer) >= 3:
        rows, method = per_analyzer, "analyzers"
    else:
        rng = np.random.default_rng(seed)
        rows = [(None, aucs(rng.integers(0, len(table), len(table)))) for _ in range(num_resamples)]
        method = "resamples"
    collected, sources = {}, {}
    for analyzer, values in rows:
        for metric_name, value in values.items():
            collected.setdefault(metric_name, []).append(value)
            sources.setdefault(metric_name, []).append(analyzer)
    # metrics with values in only a few analyzers or resamples (e.g. mostly missing) would give misleading boxes
    kept = [name for name, values in collected.items() if len(values) >= len(rows) / 2]
    return {name: collected[name] for name in kept}, {name: sources[name] for name in kept}, method


def make_metric_auc(targets):
    table = labelled_metrics_table(targets)
    if table is None or len(table) < 6:
        raise NoData("Curate or import labels to see how well each metric separates your labels.")
    metric_names = [column for column in table.columns if column not in ("analyzer", "label")]
    tasks = [("Noise vs neural", "noise"), ("SUA vs all", "good")]
    panels = []
    for task_name, positive_label in tasks:
        values, sources, method = metric_auc_values(table, positive_label, metric_names)
        if len(values) > 0:
            panels.append((task_name, values, sources, method))
    if len(panels) == 0:
        raise NoData("Label at least 3 noise and 3 other units, or 3 good and 3 other units, to compute the AUC-ROC.")

    num_metrics = max(len(values) for _, values, _, _ in panels)
    figure = new_figure(4.8 * len(panels) + 0.6, 1.2 + 0.24 * num_metrics)
    axes = figure.subplots(1, len(panels), squeeze=False)[0]
    for ax, (task_name, values, sources, method) in zip(axes, panels):
        # the metrics which separate the labels best (furthest from chance) at the top
        order = sorted(values, key=lambda name: abs(np.median(values[name]) - 0.5))
        colours = [METRIC_GROUP_COLOURS[metric_group(name)] for name in order]
        draw_boxes(ax, [values[name] for name in order], list(range(len(order))), colours, horizontal=True,
                   width=0.6, dots=False)
        if method == "analyzers":
            # one dot per analyzer, labelled with the analyzer's number (listed at the bottom)
            for position, name in enumerate(order):
                for value, analyzer in zip(values[name], sources[name]):
                    number = data_number(analyzer)
                    ax.scatter(value, position, marker=f"${number}$", s=40, color="#1D2939", zorder=4, linewidths=0)
        ax.axvline(0.5, color="#D92D20", linestyle="--", linewidth=1.2)
        ax.set_yticks(range(len(order)), [short(name, 28) for name in order], fontsize=7.5)
        ax.set_xlim(0, 1)
        ax.set_xlabel("AUC-ROC of the metric alone")
        ax.grid(axis="x", alpha=0.25)
        ax.tick_params(axis="y", length=0)
        number = len(next(iter(values.values())))
        ax.set_title(f"{task_name}\n(per analyzer, {number} analyzers)" if method == "analyzers"
                     else f"{task_name}\n({number} resamples of your units)")
    handles = [matplotlib.patches.Patch(color=colour, alpha=0.6, label=METRIC_GROUP_NAMES[group])
               for group, colour in METRIC_GROUP_COLOURS.items()]
    handles.append(matplotlib.lines.Line2D([0], [0], color="#D92D20", linestyle="--", label="Chance level"))
    figure.legend(handles=handles, loc="outside lower center", ncol=4, fontsize=8)
    return figure, []


def make_metric_correlations(targets):
    tables = []
    for target in targets:
        folder = Path(target["save_folder"])
        for file_name in ["all_metrics.csv", "labelled_metrics.csv"]:
            if (folder / file_name).is_file():
                tables.append(pd.read_csv(folder / file_name, index_col=0))
                note_data(target["path"], target_group(target), len(tables[-1]))
                break
    if len(tables) == 0:
        raise NoData("Curate or import labels to see how the metrics of your units are related.")
    metrics = pd.concat(tables).select_dtypes(include="number").replace([np.inf, -np.inf], np.nan)
    metrics = metrics.drop(columns=[column for column in ["quality", "unit_id", "index"] if column in metrics.columns])
    metrics = metrics.loc[:, (metrics.notna().sum() >= 10) & (metrics.nunique() > 1)]
    if metrics.shape[1] < 3:
        raise NoData("Not enough metrics to compare (at least 3 with 10 or more values are needed).")

    correlations = metrics.corr(method="spearman")
    # similar metrics next to each other: order them by hierarchical clustering of 1 - |correlation|
    from scipy.cluster.hierarchy import leaves_list, linkage
    from scipy.spatial.distance import squareform
    # a copy: with newer pandas, .values can be read-only
    distances = (1 - correlations.abs()).fillna(1).to_numpy(copy=True)
    np.fill_diagonal(distances, 0)
    distances = (distances + distances.T) / 2
    order = leaves_list(linkage(squareform(distances, checks=False), method="average"))
    correlations = correlations.iloc[order, order]

    size = min(11, 3 + 0.2 * len(correlations))
    figure = new_figure(size + 1, size)
    ax = figure.add_subplot()
    image = ax.imshow(correlations.values, cmap="RdBu_r", vmin=-1, vmax=1)
    labels = [short(name, 26) for name in correlations.columns]
    font_size = 8 if len(labels) <= 30 else 6.5
    ax.set_xticks(range(len(labels)), labels, rotation=90, fontsize=font_size)
    ax.set_yticks(range(len(labels)), labels, fontsize=font_size)
    ax.spines[:].set_visible(False)
    ax.tick_params(length=0)
    colour_bar = figure.colorbar(image, ax=ax, shrink=0.6, aspect=30)
    colour_bar.set_label("Spearman correlation")
    colour_bar.outline.set_visible(False)
    ax.set_title(f"How the metrics are related ({len(metrics)} units)")
    return figure, []


def make_metric_map(targets):
    from sklearn.decomposition import PCA
    from sklearn.preprocessing import StandardScaler

    table = labelled_metrics_table(targets)
    if table is None or len(table) < 6:
        raise NoData("Curate or import labels to see your units in metric space.")
    metrics = table.drop(columns=["analyzer", "label"]).replace([np.inf, -np.inf], np.nan)
    metrics = metrics.loc[:, (metrics.isna().mean() <= 0.5) & (metrics.nunique() > 1)]
    if metrics.shape[1] < 2:
        raise NoData("Not enough metrics with values to make the map.")
    metrics = metrics.fillna(metrics.median())
    pca = PCA(n_components=2)
    components = pca.fit_transform(StandardScaler().fit_transform(metrics.values))

    figure = new_figure(6.5, 5)
    ax = figure.add_subplot()
    classes = ordered_classes(table["label"].unique())
    for class_index, class_name in enumerate(classes):
        is_class = (table["label"] == class_name).values
        ax.scatter(components[is_class, 0], components[is_class, 1], s=14, alpha=0.75, linewidths=0,
                   color=class_colour(class_name, class_index), label=f"{class_name} ({is_class.sum()})")
    explained = pca.explained_variance_ratio_
    ax.set_xlabel(f"PC 1 ({explained[0]:.0%} of the variance)")
    ax.set_ylabel(f"PC 2 ({explained[1]:.0%} of the variance)")
    ax.legend(loc="best", markerscale=1.5)
    ax.set_title(f"Your units in metric space ({metrics.shape[1]} metrics, PCA)")
    return figure, []


def note_training_data(model_folder):
    """The analyzers a model was trained on (from trained_on.json), for the bottom of its figures."""
    units, _ = evaluation.read_trained_on(model_folder)
    for path, unit_ids in units.items():
        note_data(path, "training", len(unit_ids))
    note_models(Path(model_folder).name)


def make_test_confusion(model_folder):
    model_folder = Path(model_folder)
    note_training_data(model_folder)
    test_path = model_folder / TEST_PREDICTIONS_FILE
    if not test_path.is_file():
        raise NoData("The test set of this model was not saved (models trained with older versions of UnitRefine).")
    test = pd.read_csv(test_path).astype(str)
    classes = ordered_classes(get_model_classes(model_folder))
    observed = balanced_accuracy(test["human"], test["predicted"])
    resampled = bootstrap_balanced_accuracy(test["human"], test["predicted"])

    figure = new_figure(7, 3.9)
    matrix_ax, box_ax = figure.subplots(1, 2, width_ratios=[1.6, 1])
    draw_confusion(
        matrix_ax, test["human"], test["predicted"], classes, classes,
        f"{short(model_folder.name, 40)}\nbalanced accuracy {observed:.2f}  ·  {len(test)} test units",
        "Model label", "Your label",
    )
    draw_boxes(box_ax, [resampled], [0], ["#12B76A"], width=0.5, dots=False)
    box_ax.scatter([0], [observed], color="#1D2939", marker="D", s=28, zorder=4, label="test set")
    low, high = np.percentile(resampled, [2.5, 97.5])
    box_ax.set_title(f"Balanced accuracy\n95% range {low:.2f}–{high:.2f}", fontsize=9.5)
    box_ax.set_xticks([0], ["test units,\nresampled"])
    box_ax.set_ylim(0, 1.02)
    box_ax.axhline(1 / len(classes), color="#D92D20", linestyle="--", linewidth=1, label="chance")
    box_ax.grid(axis="y", alpha=0.25)
    box_ax.legend(loc="lower right", fontsize=7.5)
    return figure, []


def make_shap(model_folder, metrics_source, targets=None):
    """SHAP of a model on its own training data (`metrics_source` "training") or on the targets' units."""
    model_folder = Path(model_folder)
    classes = get_model_classes(model_folder)
    class_name = evaluation.preferred_class(classes)
    if metrics_source == "training":
        if not (model_folder / "training_data.csv").is_file():
            raise NoData("This model has no training data saved, so it can't be explained.")
        note_training_data(model_folder)
        metrics = evaluation.training_metrics(model_folder)
        where = "the units it was trained on"
    else:
        note_models(model_folder.name)
        try:
            metrics = evaluation.recording_metrics(model_folder, targets)
        except ValueError:
            raise NoData(f"{model_folder.name} can't be explained on your analyzers: none of them have all the metrics "
                         "it uses (see the notes).")
        groups = {str(target["path"]): target_group(target) for target in targets}
        for path, num_units in metrics.groupby(level=0).size().items():
            note_data(path, groups[str(path)], num_units)
        where = "the units of your analyzers"
    explanation = evaluation.shap_explanation(model_folder, metrics, class_name)
    figure = new_figure(10, 0.9 + 0.36 * min(12, len(explanation.feature_names)))
    draw_shap(figure, explanation, class_name)
    figure.suptitle(f"{model_folder.name}: on {len(explanation.values)} of {where}", x=0.01, ha="left", fontsize=10, fontweight="bold")
    return figure, []


def training_specs(project_folder, targets, trained_model_folders):
    """The figures of section 2: the labels and metrics of the curated units, and the models trained on them."""
    folder = Path(project_folder) / FIGURES_FOLDER / "2_labels_and_training"
    project_targets = [target for target in targets if target["analyzer_index"] is not None]
    params = json.dumps(sorted(str(target["path"]) for target in project_targets))
    analyzers = tuple(str(target["path"]) for target in targets)

    def label_inputs():
        return project_label_files(project_targets)

    specs = [
        FigureSpec(
            "label_counts", "Your labels", folder / "label_counts.png",
            lambda: make_label_counts(project_targets), label_inputs, params=params, analyzers=analyzers,
            caption="The share of each analyzer's units with each of your labels, as box plots with one dot per "
                    "analyzer, so the plot builds up as you add analyzers. Below each label is its total number of "
                    "units. Very unequal numbers (class imbalance) make it harder for a model to learn the rare label, "
                    "which is why SUA vs all models are recommended.",
        ),
        FigureSpec(
            "metric_distributions", "Metrics by label", folder / "metric_distributions.png",
            lambda: make_metric_distributions(project_targets), label_inputs, params=params, analyzers=analyzers,
            caption="Box plots of each metric for each of your labels (each dot is a unit), for the 9 metrics which "
                    "differ most between the labels, ranked by the effect size ε² of a Kruskal–Wallis test (0: no "
                    "difference, 1: the labels are fully separated). Boxes show the median and quartiles, whiskers "
                    "1.5 times the interquartile range. The axes show the 1st–99th percentile of the values.",
        ),
        FigureSpec(
            "metric_auc", "Metric AUC-ROC", folder / "metric_auc_roc.png",
            lambda: make_metric_auc(project_targets), label_inputs, params=params, analyzers=analyzers,
            caption="As in the UnitRefine paper: how well each metric <b>on its own</b> separates noise from neural "
                    "units, and SUA from all other units. 0.5 (red line) is chance; values far from it, towards 1 "
                    "(higher for that label) or 0 (lower for that label), mean the metric separates the labels well. "
                    "Metrics are coloured by group and sorted with the most useful at the top. Each box has one AUC "
                    "per analyzer, as in the paper; with fewer than 3 analyzers, it shows the AUC on 200 resamples of "
                    "your units instead.",
        ),
        FigureSpec(
            "metric_correlations", "Metric correlations", folder / "metric_correlations.png",
            lambda: make_metric_correlations(project_targets), label_inputs, params=params, analyzers=analyzers,
            caption="Spearman correlation between every pair of metrics, over all units of your analyzers. Metrics "
                    "are ordered so that related ones are next to each other. Strongly correlated metrics carry "
                    "similar information, so a model can rely on either of them.",
        ),
        FigureSpec(
            "metric_map", "Units in metric space", folder / "metric_map.png",
            lambda: make_metric_map(project_targets), label_inputs, params=params, analyzers=analyzers,
            caption="Each dot is a labelled unit, placed by the first two principal components of its (standardised) "
                    "metrics. Labels which form separate clouds are easy to tell apart. Overlap here doesn't mean a "
                    "model will fail, since it uses all metrics together, not only these two directions.",
        ),
    ]

    for model_folder in trained_model_folders:
        model_folder = Path(model_folder)

        def model_inputs(model_folder=model_folder):
            return [model_folder / name for name in ["best_model.skops", "model_info.json", "training_data.csv", TEST_PREDICTIONS_FILE]]

        if (model_folder / TEST_PREDICTIONS_FILE).is_file():
            specs.append(FigureSpec(
                f"test_{model_folder.name}", f"Test set: {model_folder.name}", folder / f"test_set__{short_file_name(model_folder.name)}.png",
                lambda model_folder=model_folder: make_test_confusion(model_folder), model_inputs, analyzers=analyzers,
                caption="How the model labels the units which were held out from training (the test set). Rows are "
                        "your labels, columns the model's; each cell shows the number of units and the share of its "
                        "row. A good model has most units on the diagonal. Balanced accuracy is the average of the "
                        "diagonal percentages, so every label counts equally however many units it has. "
                        "<b>Right</b>: the balanced accuracy on 500 resamples of the test units (box plot), which shows "
                        "how much the score could change with another test set of the same size; the diamond is the "
                        "test set itself, the red line chance.",
            ))
        specs.append(FigureSpec(
            f"shap_{model_folder.name}", f"SHAP: {model_folder.name}", folder / f"shap__{short_file_name(model_folder.name)}.png",
            lambda model_folder=model_folder: make_shap(model_folder, "training"), model_inputs, slow=True, analyzers=analyzers,
            caption=shap_caption(),
        ))
    return specs


def shap_caption():
    return ("<b>Left</b>: how much each metric affects the model's decision, on average. <b>Right</b>: each dot is a "
            "unit. Dots to the right of the line push the model towards the label in the title, dots to the left "
            "away from it. The colour is the metric's value (red high, blue low): e.g. red dots on the right mean "
            "high values of that metric make the label more likely.")


# --------------------------------------------------------------------------------------------------
# section 3: the predictions of the chosen models
# --------------------------------------------------------------------------------------------------

def selection_runs(mode, model_folders, key):
    """
    How a selection is predicted: a cascade as a whole, binary models one by one (so a model which can't
    label an analyzer doesn't stop the others). Each run is (mode, model folders, key, label column).
    """
    if mode == "cascade":
        return [(mode, list(model_folders), key, "cascade_model_label")]
    return [("binary", [model_folder], Path(model_folder).name, f"{Path(model_folder).name}_model_label") for model_folder in model_folders]


def selection_inputs(targets, model_folders, keys):
    """The files figures of a model selection are made from: models, labels, metrics and predictions."""
    files = []
    for model_folder in model_folders:
        model_folder = Path(model_folder)
        files += [model_folder / "best_model.skops", model_folder / "model_info.json", model_folder / evaluation.TRAINED_ON_FILE]
    for target in targets:
        save_folder = Path(target["save_folder"])
        prefix = target["file_prefix"]
        files += [save_folder / f"{prefix}labels.csv", save_folder / f"{prefix}all_metrics.csv"]
        files += sorted(save_folder.glob(f"{prefix}relabelled_units_*.csv"))
        for key in keys:
            files += [save_folder / predictions_file_name(prefix, key), Path(save_folder) / f"{prefix}units_to_relabel_{key}.csv"]
        if "://" not in str(target["path"]):
            for extension_name in ["quality_metrics", "template_metrics"]:
                extension_folder = Path(target["path"]) / "extensions" / extension_name
                if extension_folder.is_dir():
                    files += sorted(path for path in extension_folder.rglob("*") if path.is_file())
    return files


def analyzer_columns(paths):
    """The columns of a grid of panels: all analyzers together (if there are several), then each analyzer."""
    paths = list(dict.fromkeys(str(path) for path in paths))
    return ([None] if len(paths) > 1 else []) + paths


def column_title(path, groups):
    if path is None:
        return "All analyzers"
    group = "training" if groups.get(path) == "training" else "new"
    return f"[{data_number(path)}] {short(analyzer_name(path), 26)} ({group})"


def make_predicted_counts(targets, mode, model_folders, key):
    note_selection(mode, model_folders)
    runs = selection_runs(mode, model_folders, key)
    panels = []
    for run_mode, run_model_folders, run_key, label_column in runs:
        rows = []
        for target in targets:
            try:
                label_columns, _, _ = evaluation.target_predictions(target, run_mode, run_model_folders, run_key)
            except Exception as e:
                evaluation.skipped_analyzers[evaluation.skipped_key(target, run_model_folders)] = evaluation.describe_prediction_error(e)
                continue
            group = target_group(target)
            note_data(target["path"], group, len(label_columns[label_column]))
            rows.append((target["path"], label_columns[label_column].value_counts().to_dict(), group))
        panels.append((label_column.replace("_model_label", ""), rows, run_model_folders))
    if all(len(rows) == 0 for _, rows, _ in panels):
        raise NoData("The chosen model(s) could not label any of your analyzers (see the notes).")

    figure = new_figure(min(5.6 * len(panels), 14), 4.6)
    axes = figure.subplots(1, len(panels), squeeze=False)[0]
    for ax, (name, rows, run_model_folders) in zip(axes, panels):
        if len(rows) == 0:
            message_axes(ax, f"{short(name, 36)}\n\ncould not label your analyzers\n(see the notes)")
            continue
        # every class the model can give, also the ones it gave no unit
        model_classes = HUMAN_LABELS if mode == "cascade" else get_model_classes(run_model_folders[0])
        classes = ordered_classes(list(model_classes) + [class_name for _, counts, _ in rows for class_name in counts])
        draw_label_shares(ax, rows, classes)
        ax.set_title(f"Labels from {short(name, 36)}\n{len(rows)} analyzer{'s' if len(rows) > 1 else ''}")
    return figure, []


def make_confidence(targets, mode, model_folders, key):
    note_selection(mode, model_folders)
    tables = evaluation.confidence_table(targets, mode, model_folders, key)
    if mode == "binary":
        # models which could not label any analyzer are only mentioned in the notes
        for model_folder in model_folders:
            if f"{Path(model_folder).name}_model_label" not in tables:
                evaluation.skipped_analyzers.setdefault(
                    f"all analyzers (model: {Path(model_folder).name})", "it could not label any of them")
    if mode == "cascade":
        tables = {column: table for column, table in tables.items() if column == "cascade_model_label"}
    if len(tables) == 0:
        raise NoData("The chosen model(s) could not label any of your analyzers (see the notes).")
    groups = {str(target["path"]): target_group(target) for target in targets}
    for table in tables.values():
        for path, num_units in table.groupby("analyzer").size().items():
            note_data(path, groups[str(path)], num_units)
    # one row per model, one column per analyzer (and all analyzers together)
    columns = analyzer_columns(path for target in targets for path in [str(target["path"])]
                               if any((table["analyzer"] == path).any() for table in tables.values()))
    figure = new_figure(min(3.6 * len(columns), 18) + 0.4, 0.6 + 3.0 * len(tables))
    axes = figure.subplots(len(tables), len(columns), squeeze=False)
    bins = np.linspace(0, 1, 21)
    for row, (column_name, table) in enumerate(tables.items()):
        classes = ordered_classes(table["predicted"].unique())
        colours = [class_colour(name, index) for index, name in enumerate(classes)]
        for column, path in enumerate(columns):
            ax = axes[row, column]
            part = table if path is None else table[table["analyzer"] == path]
            if len(part) == 0:
                message_axes(ax, "could not label\nthis analyzer")
                continue
            ax.hist([part.loc[part["predicted"] == class_name, "confidence"] for class_name in classes], bins=bins,
                    stacked=True, label=classes, color=colours)
            ax.set_title(f"{column_title(path, groups)}\n{len(part)} units", fontsize=8.5)
            ax.set_xlim(0, 1)
            ax.set_xlabel("Model confidence")
            if column == 0:
                ax.set_ylabel(f"{short(column_name.replace('_model_label', ''), 30)}\nNumber of units")
                ax.legend(title="Model label", fontsize=7.5, title_fontsize=7.5)
    return figure, []


def make_comparison(targets, mode, model_folders, key):
    note_selection(mode, model_folders)
    evaluations = evaluation.selection_evaluations(targets, mode, model_folders, key)
    if all(result["num_candidates"] == 0 for result in evaluations):
        raise NoData("No labels to compare with yet: import labels for a new analyzer (3.3), or relabel units "
                     "(section 4). Models are never compared with labels they learned from.")
    groups = {str(target["path"]): target_group(target) for target in targets}
    for result in evaluations:
        for path, num_units in result["comparison"].groupby("analyzer").size().items():
            note_data(path, groups[str(path)], num_units)
    # one row per model, one column per analyzer with labels to compare (and all analyzers together)
    compared = set(str(path) for result in evaluations for path in result["comparison"]["analyzer"])
    columns = analyzer_columns(str(target["path"]) for target in targets if str(target["path"]) in compared)
    if len(columns) == 0:
        columns = [None]
    heights = [3.5 if len(result["comparison"]) > 0 else 0.9 for result in evaluations]
    figure = new_figure(min(3.7 * len(columns), 18) + 0.4, 0.6 + sum(heights))
    axes = figure.subplots(len(evaluations), len(columns), squeeze=False, height_ratios=heights)
    notes = []
    for row, result in enumerate(evaluations):
        comparison, classes = result["comparison"], ordered_classes(result["classes"])
        for column, path in enumerate(columns):
            ax = axes[row, column]
            part = comparison if path is None else comparison[comparison["analyzer"].astype(str) == path]
            if len(comparison) == 0:
                message_axes(ax, (f"{short(result['name'], 40)}: " + (
                    "could not label your analyzers (see the notes)" if result["num_candidates"] > 0
                    else "no labels it was not trained on")) if column == 0 else "")
                if column == 0:
                    ax.texts[-1].set_ha("left")
                    ax.texts[-1].set_x(0)
                continue
            if len(part) == 0:
                message_axes(ax, f"{column_title(path, groups)}\n\nno labels to compare\nfor this model")
                continue
            draw_confusion(
                ax, part["human"], part["predicted"], classes, classes,
                f"{column_title(path, groups)}\nbalanced accuracy {balanced_accuracy(part['human'], part['predicted']):.2f}"
                f"  ·  {len(part)} units",
                "Model label", f"{short(result['name'], 30)}\nYour label" if column == 0 else "Your label",
            )
        left_out = f" {result['num_left_out']} units were left out." if result["num_left_out"] > 0 else ""
        notes.append(f"{result['name']}: {STATUS_TEXT[result['status']]}.{left_out}")
    return figure, notes


def apply_specs(project_folder, targets, mode, model_folders, key):
    """The figures of section 3, for the chosen model selection."""
    folder = Path(project_folder) / FIGURES_FOLDER / "3_predictions" / short_file_name(key)
    model_folders = [Path(model_folder) for model_folder in model_folders]
    keys = [run[2] for run in selection_runs(mode, model_folders, key)] + [key]
    params = json.dumps([mode, [str(model_folder) for model_folder in model_folders], key, sorted(str(target["path"]) for target in targets)])
    analyzers = tuple(str(target["path"]) for target in targets)

    def inputs():
        return selection_inputs(targets, model_folders, keys)

    specs = [
        FigureSpec(
            "predicted_counts", "Predicted labels", folder / "predicted_labels.png",
            lambda: make_predicted_counts(targets, mode, model_folders, key), inputs, params=params, analyzers=analyzers,
            caption="The share of each analyzer's units which the chosen model(s) give each label, as box plots with "
                    "one marker per analyzer (circles: analyzers used for training, triangles: new analyzers), so the "
                    "plot builds up as you add analyzers. A cascade gives good, MUA or noise; a model used on its own gives its own "
                    "two classes. Below each label is its total number of units.",
        ),
        FigureSpec(
            "confidence", "Confidence", folder / "confidence.png",
            lambda: make_confidence(targets, mode, model_folders, key), inputs, params=params, analyzers=analyzers,
            caption="How sure the model is about each unit, coloured by the label it gives: all analyzers together, "
                    "then each analyzer (numbered as at the bottom of the figure). Most units should be on the right. "
                    "The units on the left are the uncertain ones, which are worth checking yourself: section 4 finds "
                    "and relabels them.",
        ),
        FigureSpec(
            "comparison", "Compared with your labels", folder / "compared_with_your_labels.png",
            lambda: make_comparison(targets, mode, model_folders, key), inputs, params=params, analyzers=analyzers,
            caption="The model's labels against your labels (curated, imported and relabelled), only for units the "
                    "model was <b>not</b> trained on, so the result is honest. One row per model; the first column has "
                    "all analyzers together, the others one analyzer each (numbered as at the bottom of the figure). "
                    "Your labels are converted to the model's classes (e.g. good and MUA count as neural). Rows are "
                    "your labels; each cell shows the number of units and the share of its row.",
        ),
    ]
    for model_folder in model_folders:
        specs.append(FigureSpec(
            f"shap_{model_folder.name}", f"SHAP: {model_folder.name}", folder / f"shap__{short_file_name(model_folder.name)}.png",
            lambda model_folder=model_folder: make_shap(model_folder, "recordings", targets), inputs, params=params, slow=True,
            analyzers=analyzers,
            caption=f"Why {model_folder.name} labels the units of your analyzers the way it does. " + shap_caption(),
        ))
    return specs


# --------------------------------------------------------------------------------------------------
# section 4: active learning
# --------------------------------------------------------------------------------------------------

def selection_rounds(mode, model_folders, key, lineages):
    """
    The model selection of each active learning round: round 1 uses the original versions of the chosen models,
    round 2 their first retrained versions, and so on (a model which was not retrained again keeps its latest
    version). Each round's picked and relabelled units are saved under its selection's key.
    Returns [(round number, model folders, key)], always including the chosen selection (`key`).
    """
    model_folders = [Path(model_folder) for model_folder in model_folders]
    versions = []
    for model_folder in model_folders:
        lineage = [model_folder]
        for folders in lineages.values():
            folders = [Path(folder) for folder in folders]
            if model_folder.name in [folder.name for folder in folders]:
                lineage = folders
        versions.append(lineage)
    rounds, keys = [], []
    for index in range(max(len(lineage) for lineage in versions)):
        folders = [lineage[min(index, len(lineage) - 1)] for lineage in versions]
        round_key = selection_key(mode, [folder.name for folder in folders])
        if round_key not in keys:
            keys.append(round_key)
            rounds.append((len(rounds) + 1, folders, round_key))
    if key not in keys:
        rounds.append((len(rounds) + 1, model_folders, key))
    return rounds


def round_title(number, mode, folders):
    names = [Path(folder).name for folder in folders]
    if mode == "cascade":
        return f"Round {number}: cascade of {short(names[0], 30)} and {short(names[1], 30)}"
    return f"Round {number}: {short(', '.join(names), 64)}"


def round_relabels(target, round_key):
    """The units relabelled in a round, as {unit id: label}."""
    path = relabelled_units_path(target, round_key)
    if not path.is_file():
        return {}
    relabels = pd.read_csv(path).dropna(subset=["quality"])
    return dict(zip(relabels["unit_id"].astype(str), relabels["quality"]))


def make_selection_confidence(targets, mode, model_folders, key, rounds):
    """
    For each round: the confidence the units were ranked by in Find least confident units, and which units were
    picked in that round, which were labelled before it, and which had no label yet.
    """
    note_selection(mode, model_folders)
    labelled_before = {str(target["path"]): set(evaluation.original_labels(target)) for target in targets}
    panels = []
    for number, folders, round_key in rounds:
        picked_anywhere = any(units_to_relabel_path(target, round_key).is_file() for target in targets)
        if picked_anywhere or round_key == key:
            rows, num_relabelled = [], 0
            for target in targets:
                try:
                    evaluation.report_progress(f"Round {number}: reading the predictions of {Path(str(target['path'])).name}")
                    table = load_or_compute_predictions(target, mode, folders, round_key)
                except Exception as e:
                    evaluation.skipped_analyzers[evaluation.skipped_key(target, folders)] = evaluation.describe_prediction_error(e)
                    continue
                note_data(target["path"], target_group(target), len(table))
                picked = set()
                if units_to_relabel_path(target, round_key).is_file():
                    picked = set(pd.read_csv(units_to_relabel_path(target, round_key))["unit_id"].astype(str))
                num_relabelled += len(set(round_relabels(target, round_key)) & picked)
                before = labelled_before[str(target["path"])]
                for unit_id, confidence in zip(table.index.astype(str), selection_confidence_from_table(table, mode)):
                    status = "before" if unit_id in before else ("picked" if unit_id in picked else "other")
                    rows.append((confidence, status))
            if len(rows) > 0:
                panels.append((number, folders, round_key, pd.DataFrame(rows, columns=["confidence", "status"]).dropna(), num_relabelled))
        # the units relabelled in this round count as labelled in the next rounds
        for target in targets:
            labelled_before[str(target["path"])] |= set(round_relabels(target, round_key))
    if len(panels) == 0:
        raise NoData("The chosen model(s) could not label any of your analyzers (see the notes).")

    figure = new_figure(7.5, 0.6 + 3.2 * len(panels))
    axes = figure.subplots(len(panels), 1, squeeze=False, sharex=True)[:, 0]
    groups = [
        ("other", "#D0D5DD", "not labelled yet"),
        ("picked", "#F04438", "picked in this round"),
        ("before", "#6172F3", "labelled before this round"),
    ]
    notes = []
    for ax, (number, folders, round_key, table, num_relabelled) in zip(axes, panels):
        ax.hist([table.loc[table["status"] == status, "confidence"] for status, _, _ in groups], bins=np.linspace(0, 1, 26),
                stacked=True, color=[colour for _, colour, _ in groups],
                label=[f"{label} ({(table['status'] == status).sum()})" for status, _, label in groups])
        picked = table[table["status"] == "picked"]
        if len(picked) > 0:
            ax.axvline(picked["confidence"].max(), color="#F04438", linestyle="--", linewidth=1)
        ax.set_xlim(0, 1)
        ax.set_ylabel("Number of units")
        ax.legend(fontsize=8)
        ax.set_title(f"{round_title(number, mode, folders)}\n{len(picked)} units picked, {num_relabelled} of them relabelled",
                     fontsize=9.5)
        if len(picked) == 0 and round_key == key:
            notes.append(f"Round {number}: click Find least confident units to pick the units to relabel.")
    axes[-1].set_xlabel("Confidence of the round's model(s)" + (" (cascade: layer 1 × layer 2)" if mode == "cascade" else
                                                               " (lowest over the models)" if len(model_folders) > 1 else ""))
    return figure, notes


def make_relabel_changes(targets, mode, model_folders, key, rounds):
    """
    Your relabels against the labels the model gave the same units: one row per round (and per model of a binary
    selection), with all analyzers together and each analyzer.
    """
    note_selection(mode, model_folders)
    groups = {str(target["path"]): target_group(target) for target in targets}
    panels = []
    for number, folders, round_key in rounds:
        runs = selection_runs(mode, folders, round_key) if mode == "binary" else [(mode, list(folders), round_key, "cascade_model_label")]
        for _, run_model_folders, _, label_column in runs:
            rows = []  # (analyzer, model label, your label)
            for target in targets:
                relabels = round_relabels(target, round_key)
                if len(relabels) == 0:
                    continue
                try:
                    # the round's predictions, which the units were picked with
                    label_columns, _, _ = evaluation.target_predictions(target, mode, folders, round_key)
                except Exception as e:
                    evaluation.skipped_analyzers[evaluation.skipped_key(target, folders)] = evaluation.describe_prediction_error(e)
                    continue
                predictions = dict(zip(label_columns[label_column].index.astype(str), label_columns[label_column].values))
                for unit_id, quality in relabels.items():
                    if unit_id in predictions:
                        rows.append((str(target["path"]), predictions[unit_id], quality))
            if len(rows) == 0:
                continue
            for path in dict.fromkeys(row[0] for row in rows):
                note_data(path, groups[path], sum(row[0] == path for row in rows))
            note_models(*[Path(folder).name for folder in run_model_folders])
            classes = get_model_classes(run_model_folders[0]) if mode == "binary" else HUMAN_LABELS
            # the row's label: the round, and the model when several binary models are compared
            name = f"Round {number}" + (f" · {short(label_column.replace('_model_label', ''), 22)}" if len(runs) > 1 else "")
            panels.append((name, ordered_classes(classes), rows))
    if len(panels) == 0:
        raise NoData("Relabel some units first: Find least confident units, then Relabel.")

    columns = analyzer_columns(str(target["path"]) for target in targets
                               if any(row[0] == str(target["path"]) for _, _, rows in panels for row in rows))
    figure = new_figure(min(3.7 * len(columns), 18) + 0.4, 0.6 + 3.4 * len(panels))
    axes = figure.subplots(len(panels), len(columns), squeeze=False)
    notes = []
    for row_index, (name, classes, rows) in enumerate(panels):
        for column, path in enumerate(columns):
            ax = axes[row_index, column]
            part = rows if path is None else [row for row in rows if row[0] == path]
            if len(part) == 0:
                message_axes(ax, f"{column_title(path, groups)}\n\nno relabelled units")
                continue
            model_labels = [row[1] for row in part]
            your_labels = [row[2] for row in part]
            # compared in the model's classes (e.g. MUA counts as neural); labels the model doesn't use
            # (e.g. noise for a SUA vs MUA model) can't agree or disagree with it
            model_classes_of_yours = [map_quality_to_model_class(label, classes) for label in your_labels]
            num_comparable = sum(mapped is not None for mapped in model_classes_of_yours)
            num_disagree = sum(mapped is not None and mapped != model_label
                               for model_label, mapped in zip(model_labels, model_classes_of_yours))
            draw_confusion(ax, model_labels, your_labels, classes, HUMAN_LABELS,
                           f"{column_title(path, groups)}\nyou disagreed on {num_disagree} of {num_comparable} units",
                           "Your label", f"{short(name, 30)}\nModel label" if column == 0 else "Model label")
            if path is None or len(columns) == 1:
                not_comparable = len(model_labels) - num_comparable
                notes.append(f"{name}: you gave {num_disagree} of {num_comparable} relabelled units a different label "
                             "from the model's (compared in the model's classes)."
                             + (f" {not_comparable} units have a label this model doesn't use." if not_comparable > 0 else ""))
    return figure, notes


def make_rounds(targets, lineages):
    lineages = {name: folders for name, folders in lineages.items() if len(folders) > 1}
    if len(lineages) == 0:
        raise NoData("The chosen model(s) have not been retrained yet. Find, relabel and retrain, and this figure "
                     "compares the versions.")
    figure = new_figure(7.5, 4)
    ax = figure.add_subplot()
    notes = []
    max_rounds = max(len(folders) for folders in lineages.values())
    width = 0.7 / len(lineages)
    analyzer_groups = {str(target["path"]): target_group(target) for target in targets}
    for index, (base_name, folders) in enumerate(lineages.items()):
        results, statuses = evaluation.lineage_evaluations(targets, folders)
        note_models(*[Path(folder).name for folder in folders])
        for path, num_units in results[0]["comparison"].groupby("analyzer").size().items():
            note_data(path, analyzer_groups[str(path)], num_units)
        colour = f"C{index}"
        offset = (index - (len(lineages) - 1) / 2) * width
        positions = [round_number + offset for round_number in range(len(folders))]
        observed, groups = [], []
        for result in results:
            comparison = result["comparison"]
            if len(comparison) == 0:
                observed.append(np.nan)
                groups.append([])
            else:
                observed.append(balanced_accuracy(comparison["human"], comparison["predicted"]))
                groups.append(bootstrap_balanced_accuracy(comparison["human"], comparison["predicted"]))
        draw_boxes(ax, groups, positions, [colour] * len(groups), width=width * 0.8, dots=False)
        num_units = len(results[0]["comparison"])
        ax.plot(positions, observed, marker="D", markersize=5, color=colour, linewidth=1.2,
                label=f"{short(base_name, 40)} ({num_units} units)", zorder=4)
        for x, accuracy in zip(positions, observed):
            if not np.isnan(accuracy):
                ax.annotate(f"{accuracy:.2f}", (x, accuracy), textcoords="offset points", xytext=(0, 9), ha="center", fontsize=8)
        if num_units == 0:
            notes.append(f"{base_name}: no labels are left that none of its versions were trained on, so the versions "
                         "can't be compared fairly. Import labels for a new analyzer (3.3) to compare them.")
        if "unknown" in statuses:
            notes.append(f"{base_name}: {STATUS_TEXT['unknown']}.")
    ax.set_xticks(range(max_rounds), ["original"] + [f"round {round_number}" for round_number in range(1, max_rounds)])
    ax.set_xlim(-0.6, max_rounds - 0.4)
    ax.set_ylim(0, 1.08)
    ax.set_ylabel("Balanced accuracy")
    ax.grid(axis="y", alpha=0.3)
    ax.legend(loc="lower right")
    ax.set_title("Accuracy over retraining rounds")
    return figure, notes


def active_learning_specs(project_folder, targets, mode, model_folders, key, lineages):
    """The figures of section 4, for the chosen model selection and its retrained versions (`lineages`)."""
    folder = Path(project_folder) / FIGURES_FOLDER / "4_active_learning" / short_file_name(key)
    model_folders = [Path(model_folder) for model_folder in model_folders]
    lineage_folders = [Path(model_folder) for folders in lineages.values() for model_folder in folders]
    rounds = selection_rounds(mode, model_folders, key, lineages)
    keys = [key] + [model_folder.name for model_folder in lineage_folders] + [round_key for _, _, round_key in rounds]
    params = json.dumps([mode, [str(model_folder) for model_folder in model_folders], key,
                         sorted(str(target["path"]) for target in targets), {name: [str(f) for f in folders] for name, folders in lineages.items()}])

    analyzers = tuple(str(target["path"]) for target in targets)

    def inputs():
        return selection_inputs(targets, list(dict.fromkeys(model_folders + lineage_folders)), keys)

    return [
        FigureSpec(
            "selection_confidence", "Units to relabel", folder / "units_to_relabel.png",
            lambda: make_selection_confidence(targets, mode, model_folders, key, rounds), inputs, params=params, analyzers=analyzers,
            caption="One panel per round of active learning (round 1: the original models, round 2: their first "
                    "retrained versions, …): all units of your analyzers, by how confident that round's model(s) were "
                    "about them. Red: the least confident units picked in that round (up to the dashed line). Blue: units "
                    "labelled before that round (curated, imported or relabelled in an earlier round), which are never "
                    "picked again, so each round asks about new units. A new panel is added with each round.",
        ),
        FigureSpec(
            "relabel_changes", "What you relabelled", folder / "what_you_relabelled.png",
            lambda: make_relabel_changes(targets, mode, model_folders, key, rounds), inputs, params=params, analyzers=analyzers,
            caption="For the units you relabelled in each round: the label that round's model gave them (rows) against "
                    "yours (columns), one row of panels per round, with all analyzers together and each analyzer. Units "
                    "where you disagree with the model teach it the most when you retrain. Units you gave the same label "
                    "as their original curated label are not saved as relabelled, so they are not counted here.",
        ),
        FigureSpec(
            "rounds", "Accuracy over rounds", folder / "accuracy_over_rounds.png",
            lambda: make_rounds(targets, lineages), inputs, params=params, analyzers=analyzers,
            caption="Balanced accuracy of the original model and each retrained version (diamonds), all compared with "
                    "the same labels of yours: the ones <b>none</b> of the versions were trained on. Each box shows the "
                    "balanced accuracy on 500 resamples of those units: if the boxes of two rounds overlap a lot, the "
                    "difference between them may be chance. If the line goes up, active learning is helping; when it "
                    "levels off, the model is ready and more rounds won't improve it much.",
        ),
    ]
