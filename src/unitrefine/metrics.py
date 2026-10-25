"""
Computes the metrics a model needs which an analyzer doesn't have yet, e.g. the drift metrics or the multi-channel
template metrics (spread, velocity_above, ...) used by the models on the Hugging Face Hub. Used by
unitrefine-apply --compute_missing_metrics (see cli.py), and by Apply in the GUI, which runs this file in its own
process (python -m unitrefine.metrics, see main below) so the window keeps responding.

The metrics are computed with SpikeInterface's default parameters, as the published models were trained with, and
saved in the analyzer. The metrics it already has are kept.
"""

import argparse
import json
import sys
import traceback

from unitrefine.import_labels import compute_all_metrics
from unitrefine.predictions import RENAMED_TEMPLATE_METRICS

# printed by main, so the GUI can show which analyzer is being done, and read the results
METRICS_STEP_MARKER = "UNITREFINE_METRICS_STEP:"
METRICS_RESULT_MARKER = "UNITREFINE_METRICS_RESULT:"

# the extensions metrics can depend on, in the order they can be computed in
PARENT_EXTENSIONS = ["random_spikes", "waveforms", "templates", "noise_levels", "spike_amplitudes", "spike_locations",
                     "principal_components"]


def required_metric_columns(model_folders):
    """The metric columns the models use, with the names SpikeInterface uses now (e.g. peak_to_valley is now
    peak_to_trough_duration)."""
    from spikeinterface.curation.model_based_curation import get_required_metrics_from_model

    columns = set()
    for model_folder in model_folders:
        columns |= set(get_required_metrics_from_model(model_folder=model_folder, trust_model=True))
    return {RENAMED_TEMPLATE_METRICS.get(column, (column,))[0] for column in columns}


def metrics_to_compute(analyzer, model_folders):
    """
    The metrics to compute so the models can be applied: {"quality_metrics" or "template_metrics": [metric names]},
    empty if the analyzer has them all.
    """
    from spikeinterface.metrics import ComputeQualityMetrics, ComputeTemplateMetrics

    columns = set(compute_all_metrics(analyzer).columns)
    # metrics computed before SpikeInterface 0.103.2 have the old names (e.g. peak_to_valley), which models accept
    columns |= {new_name for old_name, (new_name, _) in RENAMED_TEMPLATE_METRICS.items() if old_name in columns}
    missing = required_metric_columns(model_folders) - columns
    to_compute = {}
    for extension in [ComputeQualityMetrics, ComputeTemplateMetrics]:
        for metric in extension.metric_list:
            if len(missing & set(metric.metric_columns)) > 0:
                to_compute.setdefault(extension.extension_name, []).append(metric.metric_name)
                missing -= set(metric.metric_columns)
    if len(missing) > 0:
        raise ValueError(f"SpikeInterface can't compute the metrics {sorted(missing)}, which the models use.")
    return to_compute


def compute_missing_metrics(analyzer, model_folders, job_kwargs=None):
    """
    Computes and saves the metrics the models need which `analyzer` doesn't have, and the extensions they
    depend on. Returns {extension: [metric names]} of what was computed.
    """
    from spikeinterface.metrics import ComputeQualityMetrics, ComputeTemplateMetrics, get_multi_channel_template_metric_names

    to_compute = metrics_to_compute(analyzer, model_folders)
    if len(to_compute) == 0:
        return to_compute

    needed = set()
    if "template_metrics" in to_compute:
        needed |= {"random_spikes", "templates"}
    if "quality_metrics" in to_compute:
        needed |= set(ComputeQualityMetrics.get_optional_dependencies(metric_names=to_compute["quality_metrics"]))
        # e.g. "amplitude_scalings" or "spike_amplitudes": spike_amplitudes is used
        needed -= {"amplitude_scalings"}
    if "principal_components" in needed:
        needed |= {"waveforms"}
    if len(needed) > 0:
        needed |= {"random_spikes"}
    parents = [name for name in PARENT_EXTENSIONS if name in needed and not analyzer.has_extension(name)]
    if len(parents) > 0 and not (analyzer.has_recording() or analyzer.has_temporary_recording()):
        raise ValueError(f"The analyzer has no recording, which is needed to compute {', '.join(parents)} for the "
                         f"metrics {to_compute}. Set its recording, or compute these metrics where the recording is.")

    for name in parents:
        compute_keeping_children(analyzer, name, job_kwargs)
    for extension_name, metric_names in to_compute.items():
        print(f"    Computing {', '.join(metric_names)}...", flush=True)
        params = dict(metric_names=list(metric_names), delete_existing_metrics=False)
        if extension_name == "template_metrics":
            params["include_multi_channel_metrics"] = len(set(metric_names) & set(get_multi_channel_template_metric_names())) > 0
        analyzer.compute(extension_name, **params, **(job_kwargs or {}))
    return to_compute


def compute_keeping_children(analyzer, extension_name, job_kwargs=None):
    """
    Computes an extension. SpikeInterface deletes the extensions which depend on it (e.g. computing waveforms, for
    the PCA metrics, deletes the templates and the template metrics), so these are computed again, with the
    parameters they had. Except for unit_locations, whose fit can change a little, they come back the same.
    """
    from spikeinterface.core.sortinganalyzer import _get_children_dependencies

    children = [child for child in dict.fromkeys(_get_children_dependencies(extension_name)) if analyzer.has_extension(child)]
    children_params = {child: analyzer.get_extension(child).params for child in children}
    print(f"    Computing {extension_name}...", flush=True)
    analyzer.compute(extension_name, **(job_kwargs or {}))
    for child in children:
        if not analyzer.has_extension(child):
            print(f"    Computing {child} again (computing {extension_name} deleted it), with the same parameters...", flush=True)
            analyzer.compute(child, **children_params[child], **(job_kwargs or {}))


def main(argv=None):
    """
    Computes the metrics the models need for each analyzer (see compute_missing_metrics). Before each analyzer it
    prints METRICS_STEP_MARKER and "analyzer_{index}"; at the end, METRICS_RESULT_MARKER and a json list with, per
    analyzer, its "path" and the "computed" metrics or the "error".
    """
    from spikeinterface.core import load_sorting_analyzer

    parser = argparse.ArgumentParser(description="Computes the metrics models need which analyzers don't have.")
    parser.add_argument("analyzers", nargs="+")
    parser.add_argument("--model", action="append", required=True, dest="models", help="a model folder")
    parser.add_argument("--n_jobs", type=int, default=-1, help="processes to use (-1: all cores)")
    args = parser.parse_args(argv)

    results = []
    for index, path in enumerate(args.analyzers):
        print(f"{METRICS_STEP_MARKER} analyzer_{index}", flush=True)
        print(f"\nComputing the missing metrics of {path}...", flush=True)
        try:
            analyzer = load_sorting_analyzer(path, load_extensions=False)
            results.append(dict(path=path, computed=compute_missing_metrics(analyzer, args.models, dict(n_jobs=args.n_jobs))))
        except Exception as error:
            traceback.print_exc()
            results.append(dict(path=path, error=str(error)))
    print(f"{METRICS_RESULT_MARKER}{json.dumps(results)}", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
