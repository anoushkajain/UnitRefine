"""
Using a Phy folder in UnitRefine: makes a SortingAnalyzer from it, and saves Phy's labels. There are two kinds:

- **Kilosort output**, curated in Phy. There is no analyzer yet, so one is made from Phy's clusters and the
  recording Kilosort sorted (found from params.py), with all the metrics the models use.
- **A Phy folder exported from an analyzer** with SpikeInterface's export_to_phy (it has cluster_si_unit_ids.tsv).
  Phy's merges and splits are applied to that analyzer (see phy_curation.py). If Phy only changed labels, the
  analyzer is used as it is.

Either way, Phy's labels (good, mua, noise; unsorted is skipped) are saved as a csv with unit_id and quality
columns, which UnitRefine imports like any other labels.

Runs without the GUI too, e.g. on a computing cluster:

    python -m unitrefine.phy_import <phy folder> [--analyzer <analyzer exported to Phy>] [--output <folder>]
"""

import argparse
import ast
import json
import sys
import time
import traceback
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd

# lines printed for the GUI, which runs this module in its own process
PHY_STEP_MARKER = "UNITREFINE_PHY_STEP:"
PHY_RESULT_MARKER = "UNITREFINE_PHY_RESULT:"

# written next to the analyzer, so it is known where it came from
MADE_FROM_PHY_FILE = "made_from_phy.json"

KILOSORT_STEPS = [
    ("start", "Starting Python and SpikeInterface"),
    ("reading", "Reading the Phy folder and the recording"),
    ("waveforms", "Extracting waveforms (the longest step)"),
    ("extensions", "Computing templates, amplitudes, locations and correlograms"),
    ("pca", "Computing principal components"),
    ("metrics", "Computing quality and template metrics"),
    ("labels", "Saving Phy's labels"),
]
EXPORTED_STEPS = [
    ("start", "Starting Python and SpikeInterface"),
    ("reading", "Reading the Phy folder and the analyzer"),
    ("curation", "Applying Phy's merges and splits"),
    ("saving", "Saving the curated analyzer"),
    ("labels", "Saving Phy's labels"),
]

# the extensions computed for an analyzer made from Kilosort output: what the models, the figures and the
# SpikeInterface-GUI use, as for an analyzer made in SpikeInterface
EXTENSION_STEPS = [
    ("waveforms", ["random_spikes", "waveforms"]),
    ("extensions", ["templates", "noise_levels", "spike_amplitudes", "unit_locations", "correlograms", "template_similarity"]),
    ("pca", ["principal_components"]),
    ("metrics", ["quality_metrics", "template_metrics"]),
]

# from the SpikeGLX documentation: µV per bit of the AP band, for the usual probes
GAIN_HINT = "e.g. 2.34 for Neuropixels 1.0 (gain 500), 0.763 for Neuropixels 2.0"


class NeedsGain(ValueError):
    """The recording is a plain binary file, so its gain (µV per bit) must be given."""


class RecordingNotFound(FileNotFoundError):
    """The recording Kilosort sorted is not where params.py says."""


def report_step(step):
    print(f"{PHY_STEP_MARKER} {step}", flush=True)


# --------------------------------------------------------------------------------------------------
# what is in a Phy folder
# --------------------------------------------------------------------------------------------------

def read_params(phy_folder):
    """The settings in a Phy folder's params.py, as a dict (read without running the file)."""
    params = {}
    tree = ast.parse((Path(phy_folder) / "params.py").read_text())
    for statement in tree.body:
        if isinstance(statement, ast.Assign) and len(statement.targets) == 1 and isinstance(statement.targets[0], ast.Name):
            try:
                params[statement.targets[0].id] = ast.literal_eval(statement.value)
            except ValueError:
                pass
    return params


def phy_folder_kind(phy_folder):
    """
    "exported" for a Phy folder made with SpikeInterface's export_to_phy, "kilosort" for Kilosort output.
    Raises ValueError if the folder is not a Phy folder.
    """
    phy_folder = Path(phy_folder)
    missing = [name for name in ["params.py", "spike_times.npy", "spike_templates.npy"] if not (phy_folder / name).is_file()]
    if len(missing) > 0:
        raise ValueError(f"{phy_folder} is not a Phy folder: it has no {', '.join(missing)}. Choose the folder with "
                         "params.py, which Kilosort writes and Phy opens.")
    return "exported" if (phy_folder / "cluster_si_unit_ids.tsv").is_file() else "kilosort"


def default_output_folder(phy_folder):
    """Where the analyzer made from a Phy folder is saved: next to it."""
    phy_folder = Path(phy_folder)
    return phy_folder.parent / f"{phy_folder.name}_analyzer"


def made_from_phy(analyzer_folder):
    """What an analyzer was made from (the contents of made_from_phy.json), or None."""
    path = Path(analyzer_folder) / MADE_FROM_PHY_FILE
    if not path.is_file():
        return None
    with open(path) as f:
        return json.load(f)


def recording_file(phy_folder, params=None):
    """The recording file Kilosort sorted, from params.py, or None if it is not found."""
    phy_folder = Path(phy_folder)
    params = read_params(phy_folder) if params is None else params
    dat_paths = params.get("dat_path")
    if dat_paths is None:
        return None
    if isinstance(dat_paths, (list, tuple)):
        if len(dat_paths) != 1:
            return None
        dat_paths = dat_paths[0]
    path = Path(dat_paths)
    # relative paths are relative to the Phy folder; the folder may also have been moved with its recording
    for candidate in [path if path.is_absolute() else phy_folder / path, phy_folder / path.name, phy_folder.parent / path.name]:
        if candidate.is_file():
            return candidate
    return None


def recording_needs_gain(path):
    """True for plain binary files, whose gain is not saved with them (SpikeGLX and IBL files have it)."""
    path = Path(path)
    return not (path.with_suffix(".meta").is_file() or path.suffix == ".cbin")


def matches_phy_folder(analyzer_or_sorting, phy_folder):
    """True if the Phy folder was exported from this analyzer (same units, same spike counts)."""
    from unitrefine.phy_curation import phy_to_curation

    try:
        phy_to_curation(phy_folder, analyzer_or_sorting)
        return True
    except ValueError:
        return False


def phy_has_edits(phy_folder):
    """True if clusters were merged or split in Phy (spike_clusters differs from the exported units)."""
    phy_folder = Path(phy_folder)
    clusters_path = phy_folder / "spike_clusters.npy"
    if not clusters_path.is_file():
        return False
    return not np.array_equal(np.load(clusters_path).ravel(), np.load(phy_folder / "spike_templates.npy").ravel())


# --------------------------------------------------------------------------------------------------
# reading the recording of Kilosort output
# --------------------------------------------------------------------------------------------------

def load_kilosort_recording(phy_folder, path=None, gain_to_uV=None):
    """
    The recording Kilosort sorted, with the channels Kilosort used, filtered as for spike sorting (unless
    params.py says it already is): high-pass filtered and common median referenced, and phase shifted for
    Neuropixels recordings from SpikeGLX. Returns (recording, description of how it was read).
    """
    import spikeinterface.full as si
    from probeinterface import Probe

    phy_folder = Path(phy_folder)
    params = read_params(phy_folder)
    path = recording_file(phy_folder, params) if path is None else Path(path)
    if path is None or not path.is_file():
        raise RecordingNotFound(
            f"The recording Kilosort sorted was not found (params.py: dat_path = {params.get('dat_path')!r}). "
            "Choose the recording file (e.g. the .ap.bin file from SpikeGLX)."
        )

    steps = []
    if path.with_suffix(".meta").is_file():
        # SpikeGLX: e.g. run_g0_t0.imec0.ap.bin is the stream imec0.ap; the gains are in the .meta file
        stream_name = ".".join(path.name.split(".")[-3:-1])
        recording = si.read_spikeglx(path.parent, stream_name=stream_name)
        steps.append(f"SpikeGLX stream {stream_name}")
    elif path.suffix == ".cbin":
        recording = si.read_cbin_ibl(cbin_file_path=path)
        steps.append("IBL compressed binary (.cbin)")
    else:
        if gain_to_uV is None:
            raise NeedsGain(f"{path.name} is a binary file which doesn't say how to convert its values to µV. "
                            f"Give its gain in µV per bit ({GAIN_HINT}).")
        recording = si.read_binary(
            path, sampling_frequency=float(params["sample_rate"]), dtype=params.get("dtype", "int16"),
            num_channels=int(params["n_channels_dat"]), file_offset=int(params.get("offset", 0)), gain_to_uV=float(gain_to_uV),
            offset_to_uV=0.0,
        )
        steps.append(f"binary file, gain {gain_to_uV} µV per bit")

    # the channels Kilosort used (e.g. without the sync channel), in its order
    channel_map = np.load(phy_folder / "channel_map.npy").ravel() if (phy_folder / "channel_map.npy").is_file() else np.arange(recording.get_num_channels())
    recording = recording.select_channels(recording.channel_ids[channel_map])
    if not recording.has_probe():
        positions = np.load(phy_folder / "channel_positions.npy")
        probe = Probe(ndim=2, si_units="um")
        probe.set_contacts(positions=positions, shapes="square", shape_params={"width": 12})
        probe.set_device_channel_indices(np.arange(len(positions)))
        recording.set_probe(probe, in_place=True)

    if not params.get("hp_filtered", False):
        if "SpikeGLX" in steps[0]:
            recording = si.phase_shift(recording)
            steps.append("phase shift")
        recording = si.highpass_filter(recording, freq_min=300.0)
        recording = si.common_reference(recording, reference="global", operator="median")
        steps += ["high-pass filter at 300 Hz", "common median reference"]
    return recording, ", ".join(steps)


# --------------------------------------------------------------------------------------------------
# making the analyzer
# --------------------------------------------------------------------------------------------------

def phy_labels(sorting):
    """Phy's labels as a table with unit_id and quality columns: good, MUA, noise (other labels are left out)."""
    from unitrefine.phy_curation import QUALITY_LABELS

    if "quality" not in sorting.get_property_keys():
        return pd.DataFrame(columns=["unit_id", "quality"])
    labels = [QUALITY_LABELS.get(str(label).strip().lower()) for label in sorting.get_property("quality")]
    table = pd.DataFrame({"unit_id": sorting.unit_ids, "quality": labels})
    return table.dropna(subset=["quality"]).reset_index(drop=True)


def kilosort_sorting(phy_folder):
    """
    Phy's clusters (after its merges and splits) as a sorting, with the cluster ids as unit ids. Made from the
    spikes, not with read_phy, which leaves out clusters missing from the cluster_*.tsv files.
    """
    import spikeinterface.full as si

    phy_folder = Path(phy_folder)
    spike_times = np.load(phy_folder / "spike_times.npy").ravel().astype("int64")
    clusters_path = phy_folder / "spike_clusters.npy"
    spike_clusters = np.load(clusters_path if clusters_path.is_file() else phy_folder / "spike_templates.npy").ravel().astype("int64")
    order = np.argsort(spike_times, kind="stable")
    return si.NumpySorting.from_samples_and_labels([spike_times[order]], [spike_clusters[order]],
                                                   float(read_params(phy_folder)["sample_rate"]))


def kilosort_phy_labels(phy_folder, unit_ids):
    """
    The labels set in Phy (cluster_group.tsv), as a table with unit_id and quality columns. Kilosort writes its
    own automatic labels to cluster_group.tsv too: if they are all the same as cluster_KSLabel.tsv, the folder
    was not labelled in Phy, so no labels are returned.
    """
    from unitrefine.phy_curation import QUALITY_LABELS

    phy_folder = Path(phy_folder)
    group_file = phy_folder / "cluster_group.tsv"
    if not group_file.is_file():
        print("No cluster_group.tsv: no units were labelled in Phy.", flush=True)
        return pd.DataFrame(columns=["unit_id", "quality"])
    groups = pd.read_csv(group_file, sep="	")
    groups = dict(zip(groups["cluster_id"].astype("int64"), groups[[column for column in groups.columns if column != "cluster_id"][0]]))
    ks_file = phy_folder / "cluster_KSLabel.tsv"
    if ks_file.is_file():
        ks_labels = pd.read_csv(ks_file, sep="	")
        ks_labels = dict(zip(ks_labels["cluster_id"].astype("int64"), ks_labels["KSLabel"]))
        if all(ks_labels.get(cluster_id) == label for cluster_id, label in groups.items()):
            print("The labels in cluster_group.tsv are Kilosort's automatic labels (the same as cluster_KSLabel.tsv), "
                  "not labels set in Phy, so they are not used.", flush=True)
            return pd.DataFrame(columns=["unit_id", "quality"])
    unit_ids = set(int(unit_id) for unit_id in unit_ids)
    rows = [(cluster_id, QUALITY_LABELS.get(str(label).strip().lower())) for cluster_id, label in groups.items() if cluster_id in unit_ids]
    return pd.DataFrame(rows, columns=["unit_id", "quality"]).dropna(subset=["quality"]).reset_index(drop=True)


def analyzer_from_kilosort(phy_folder, output_folder, recording_path=None, gain_to_uV=None, job_kwargs=None):
    """Makes an analyzer from Kilosort output curated in Phy, saved to `output_folder`. Returns (analyzer, labels, how the recording was read)."""
    import spikeinterface.full as si

    phy_folder = Path(phy_folder)
    report_step("reading")
    sorting = kilosort_sorting(phy_folder)
    recording, how_read = load_kilosort_recording(phy_folder, recording_path, gain_to_uV)
    print(f"Recording: {how_read}. {sorting.get_num_units()} units from Phy.", flush=True)

    analyzer = si.create_sorting_analyzer(sorting, recording, format="binary_folder", folder=output_folder, sparse=True,
                                          overwrite=True)
    for step, extensions in EXTENSION_STEPS:
        report_step(step)
        analyzer.compute(extensions, **(job_kwargs or {}))
    return analyzer, kilosort_phy_labels(phy_folder, sorting.unit_ids), how_read


def analyzer_from_exported(phy_folder, analyzer_path, output_folder, allow_hard_merge=False, job_kwargs=None):
    """
    Applies Phy's curation to the analyzer exported to Phy. Returns (analyzer path, labels): the analyzer itself
    if Phy only changed labels, else the curated analyzer saved to `output_folder`.
    """
    import spikeinterface.full as si
    from unitrefine.phy_curation import apply_phy_curation

    report_step("reading")
    analyzer = si.load_sorting_analyzer(analyzer_path)
    report_step("curation")
    curated = apply_phy_curation(analyzer, phy_folder, allow_hard_merge=allow_hard_merge, **(job_kwargs or {}))
    labels = phy_labels(curated.sorting)
    if not phy_has_edits(phy_folder):
        print("Phy only changed labels, so the analyzer is used as it is.", flush=True)
        return Path(analyzer_path), labels, False
    report_step("saving")
    curated.save_as(format="binary_folder", folder=output_folder)
    return Path(output_folder), labels, True


def import_phy_folder(phy_folder, output_folder=None, labels_path=None, analyzer_path=None, recording_path=None,
                      gain_to_uV=None, allow_hard_merge=False, job_kwargs=None, reuse=False):
    """
    Makes the analyzer for a Phy folder (see the module description) and saves Phy's labels to `labels_path`
    (default: next to the analyzer, as <analyzer>_phy_labels.csv). Returns a dict with the `analyzer` path,
    the `labels` path and their number, the `kind` of Phy folder, and whether a `new_analyzer` was saved.
    With `reuse`, the analyzer saved earlier in `output_folder` is used, and only the labels are read again.
    """
    phy_folder = Path(phy_folder)
    kind = phy_folder_kind(phy_folder)
    output_folder = default_output_folder(phy_folder) if output_folder is None else Path(output_folder)
    start_time = time.time()

    if reuse:
        import spikeinterface.full as si

        report_step("reading")
        sorting = si.load_sorting_analyzer(output_folder, load_extensions=False).sorting
        labels = kilosort_phy_labels(phy_folder, sorting.unit_ids) if kind == "kilosort" else phy_labels(sorting)
        analyzer_folder, new_analyzer = output_folder, False
        print(f"Using the analyzer made earlier: {output_folder}", flush=True)
    elif kind == "kilosort":
        _, labels, how_read = analyzer_from_kilosort(phy_folder, output_folder, recording_path, gain_to_uV, job_kwargs)
        analyzer_folder, new_analyzer = output_folder, True
        details = dict(recording=str(recording_path or recording_file(phy_folder)), recording_read_as=how_read)
    else:
        if analyzer_path is None:
            raise ValueError(f"{phy_folder} was exported from a SpikeInterface analyzer: give that analyzer (--analyzer).")
        analyzer_folder, labels, new_analyzer = analyzer_from_exported(phy_folder, analyzer_path, output_folder,
                                                                      allow_hard_merge, job_kwargs)
        details = dict(exported_from=str(analyzer_path))

    if new_analyzer:
        with open(output_folder / MADE_FROM_PHY_FILE, "w") as f:
            json.dump(dict(phy_folder=str(phy_folder), kind=kind, date=datetime.now().isoformat(timespec="seconds"),
                           **details), f, indent=4)

    report_step("labels")
    labels_path = Path(labels_path) if labels_path is not None else output_folder.parent / f"{output_folder.name}_phy_labels.csv"
    labels_path.parent.mkdir(parents=True, exist_ok=True)
    labels.to_csv(labels_path, index=False)
    counts = {label: int(count) for label, count in labels['quality'].value_counts().items()}
    print(f"{len(labels)} units have a Phy label {counts}. Saved in {labels_path}.", flush=True)
    print(f"Done in {time.time() - start_time:.0f} s. Analyzer: {analyzer_folder}", flush=True)
    return dict(kind=kind, analyzer=str(analyzer_folder), new_analyzer=new_analyzer, labels=str(labels_path),
                num_labels=len(labels))


# --------------------------------------------------------------------------------------------------
# command line
# --------------------------------------------------------------------------------------------------

def main(argv=None):
    from unitrefine.phy_curation import HardMergeRequired

    parser = argparse.ArgumentParser(prog="python -m unitrefine.phy_import", description=__doc__.split("\n\n")[0],
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("phy_folder", help="The Phy folder (with params.py)")
    parser.add_argument("--output", default=None, help="Where to save the analyzer (default: next to the Phy folder)")
    parser.add_argument("--labels", default=None, help="Where to save Phy's labels (csv)")
    parser.add_argument("--analyzer", default=None, help="For a Phy folder exported from an analyzer: that analyzer")
    parser.add_argument("--recording", default=None, help="For Kilosort output: the recording, if params.py doesn't find it")
    parser.add_argument("--gain_to_uv", type=float, default=None, help=f"For a plain binary recording: µV per bit ({GAIN_HINT})")
    parser.add_argument("--allow_hard_merge", action="store_true", help="Do merges which need the raw recording")
    parser.add_argument("--reuse", action="store_true", help="Use the analyzer made earlier in --output; only read the labels again")
    parser.add_argument("--n_jobs", type=float, default=-1, help="Processes to use (-1: all cores)")
    args = parser.parse_args(argv)

    report_step("start")
    try:
        result = import_phy_folder(
            args.phy_folder, output_folder=args.output, labels_path=args.labels, analyzer_path=args.analyzer,
            recording_path=args.recording, gain_to_uV=args.gain_to_uv, allow_hard_merge=args.allow_hard_merge,
            job_kwargs=dict(n_jobs=int(args.n_jobs) if args.n_jobs >= 1 or args.n_jobs == -1 else args.n_jobs),
            reuse=args.reuse,
        )
        result["status"] = "done"
    except HardMergeRequired as e:
        result = dict(status="hard_merge_required", message=str(e))
    except NeedsGain as e:
        result = dict(status="needs_gain", message=str(e))
    except RecordingNotFound as e:
        result = dict(status="needs_recording", message=str(e))
    except Exception as e:
        traceback.print_exc()
        result = dict(status="error", message=f"{type(e).__name__}: {e}")
    if result["status"] != "done":
        print(result["message"], flush=True)
    print(f"{PHY_RESULT_MARKER} {json.dumps(result)}", flush=True)
    return 0 if result["status"] == "done" else 1


if __name__ == "__main__":
    sys.exit(main())
