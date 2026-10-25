"""
Applies models to analyzers without the GUI, e.g. on a computing cluster (HPC). Nothing here imports Qt, so it
runs without a screen.

    unitrefine-apply ANALYZERS --model MODEL [--model MODEL ...] [--cascade] [--output FOLDER] [--compute_missing_metrics]
    unitrefine-download-models [REPO_ID ...] [--models_folder FOLDER]

ANALYZERS can be analyzer folders, folders with analyzers inside, or text files with one analyzer path per line.
A MODEL is a model folder (e.g. my_project/models/my_model) or a Hugging Face repo id, which is downloaded once
into the models folder. The predictions are the ones the GUI makes, with the same labels and columns: one csv per
analyzer, {output}/{analyzer name}__labels_from_{models}.csv. Running it again reuses the predictions which are up
to date, so only new or changed analyzers are labelled. With --compute_missing_metrics, the metrics the models need
which an analyzer doesn't have (e.g. the drift metrics) are computed first, and saved in the analyzer.

With --task_index, only one of the analyzers is labelled, e.g. one per job of a SLURM job array
(--task_index $SLURM_ARRAY_TASK_ID); --list shows the analyzers and their numbers. The csv files are named
from the whole list, so the jobs' files don't clash.
"""

import argparse
import os
import shutil
import sys
from pathlib import Path

from unitrefine import __version__
from unitrefine.models_hub import downloaded_repo_id, download_model, hfh_model_groups, looks_like_repo_id
from unitrefine.predictions import (
    analyzer_output_names,
    apply_to_analyzers,
    find_analyzers,
    get_model_classes,
    is_layer1_model,
    is_layer2_model,
    is_remote_path,
    selection_key,
)

# where models downloaded from the Hugging Face Hub are kept, unless --models_folder is given
MODELS_FOLDER_VARIABLE = "UNITREFINE_MODELS_FOLDER"
DEFAULT_MODELS_FOLDER = Path.home() / ".cache" / "unitrefine" / "models"


class UsageError(ValueError):
    """A problem with what was asked for, e.g. a model which doesn't exist: shown as a message, not a traceback."""


def default_models_folder():
    return Path(os.environ.get(MODELS_FOLDER_VARIABLE, DEFAULT_MODELS_FOLDER))


def collect_analyzers(sources, max_depth=3):
    """
    The analyzers in `sources`: analyzer folders, folders with analyzers inside (up to `max_depth` levels down),
    text files with one path per line (blank lines and lines starting with # are skipped), or remote (s3) paths.
    """
    analyzers = []
    for source in sources:
        source = str(source)
        if is_remote_path(source):
            found = [source]
        elif Path(source).is_file():
            lines = [line.strip() for line in Path(source).read_text().splitlines()]
            found = collect_analyzers([line for line in lines if line != "" and not line.startswith("#")], max_depth)
        elif Path(source).is_dir():
            found = find_analyzers(source, max_depth)
            if len(found) == 0:
                raise UsageError(f"No analyzers found in {source} (searched {max_depth} folder levels down).")
        else:
            raise UsageError(f"{source} does not exist.")
        for path in found:
            if str(path) not in [str(analyzer) for analyzer in analyzers]:
                analyzers.append(path)
    return analyzers


def resolve_model(model, models_folder=None, download=True):
    """
    The folder of `model`: a model folder, or a Hugging Face repo id, which is downloaded into `models_folder`
    the first time. Several jobs can download the same model at once: each downloads it to its own folder, and
    the first one to finish is kept.
    """
    if Path(model).is_dir():
        model_folder = Path(model)
        if not (model_folder / "best_model.skops").is_file():
            raise UsageError(f"{model_folder} is not a UnitRefine model: it has no best_model.skops.")
        return model_folder
    if not looks_like_repo_id(model):
        raise UsageError(f"The model {model} was not found: give a model folder, or a Hugging Face repo id "
                         f"such as SpikeInterface/UnitRefine_sua_mua_classifier.")

    models_folder = Path(models_folder) if models_folder is not None else default_models_folder()
    model_folder = models_folder / model.split("/")[-1]
    if model_folder.is_dir():
        if downloaded_repo_id(model_folder) != model:
            raise UsageError(f"{model_folder} exists, but was not downloaded from {model}. Give another --models_folder.")
        return model_folder
    if not download:
        raise UsageError(f"{model} has not been downloaded into {models_folder}.")

    models_folder.mkdir(parents=True, exist_ok=True)
    temporary_folder = models_folder / f".{model_folder.name}.download-{os.getpid()}"
    try:
        download_model(model, temporary_folder)
    except Exception as e:
        raise UsageError(
            f"Could not download {model} from the Hugging Face Hub: {e}\nIf this computer has no internet (e.g. a "
            f"compute node), download the model first where there is, with: unitrefine-download-models {model} "
            f"--models_folder {models_folder}"
        ) from e
    try:
        temporary_folder.rename(model_folder)
    except OSError:
        if not model_folder.is_dir():
            raise
        # another job downloaded it at the same time
        shutil.rmtree(temporary_folder, ignore_errors=True)
    print(f"Saved {model} in {model_folder}.")
    return model_folder


def model_selection(model_folders, cascade):
    """The mode ("binary" or "cascade") and the model folders in the order they are applied."""
    if not cascade:
        return "binary", list(model_folders)
    if len(model_folders) != 2:
        raise UsageError("A cascade needs two models: a noise vs neural model and a SUA vs MUA model.")
    first, second = model_folders
    if is_layer1_model(get_model_classes(first)) and is_layer2_model(get_model_classes(second)):
        return "cascade", [first, second]
    if is_layer1_model(get_model_classes(second)) and is_layer2_model(get_model_classes(first)):
        return "cascade", [second, first]
    raise UsageError(
        f"A cascade needs a noise vs neural model and a SUA vs MUA model, but {Path(first).name} predicts "
        f"{get_model_classes(first)} and {Path(second).name} predicts {get_model_classes(second)}."
    )


def apply_models(analyzers, models, cascade=False, output_folder="unitrefine_predictions", models_folder=None,
                 task_index=None, max_depth=3, compute_missing_metrics=False, n_jobs=1):
    """
    Applies `models` (model folders or Hugging Face repo ids) to `analyzers` (see collect_analyzers), each on its
    own or as a cascade, and saves one predictions csv per analyzer in `output_folder`. With `task_index`, only
    that analyzer (counting from 0) is labelled. With `compute_missing_metrics`, the metrics the models need which
    an analyzer doesn't have are computed first (with `n_jobs` processes) and saved in the analyzer. Returns a list of dicts with the analyzer's `path` and the saved
    `file`, or the `error` for analyzers which could not be labelled, e.g. because a metric is missing.
    """
    analyzer_paths = collect_analyzers(analyzers, max_depth)
    if len(analyzer_paths) == 0:
        raise UsageError("No analyzers were given.")
    # the names come from the whole list, so each job of an array gets a different file name
    output_names = analyzer_output_names(analyzer_paths)
    if task_index is not None:
        if not 0 <= task_index < len(analyzer_paths):
            raise UsageError(f"--task_index {task_index} is out of range: there are {len(analyzer_paths)} analyzers, "
                             f"numbered 0 to {len(analyzer_paths) - 1}.")
        analyzer_paths, output_names = [analyzer_paths[task_index]], [output_names[task_index]]

    model_folders = [resolve_model(model, models_folder) for model in models]
    mode, model_folders = model_selection(model_folders, cascade)
    key = selection_key(mode, [model_folder.name for model_folder in model_folders])
    return apply_to_analyzers(analyzer_paths, mode, model_folders, key, output_folder, output_names=output_names,
                              compute_missing_metrics=compute_missing_metrics, job_kwargs=dict(n_jobs=n_jobs))


def apply_main(argv=None):
    parser = argparse.ArgumentParser(
        prog="unitrefine-apply",
        description="Applies UnitRefine models to Sorting Analyzers and saves their predictions, one csv per analyzer.",
        epilog="Example: unitrefine-apply /data/sortings --model SpikeInterface/UnitRefine_noise_neural_classifier "
               "--model SpikeInterface/UnitRefine_sua_mua_classifier --cascade --output predictions",
    )
    parser.add_argument("analyzers", nargs="+",
                        help="analyzer folders, folders with analyzers inside, or text files with one analyzer per line")
    parser.add_argument("--version", action="version", version=f"unitrefine {__version__}")
    parser.add_argument("--model", action="append", default=[], dest="models",
                        help="a model folder or a Hugging Face repo id; give --model again for more models")
    parser.add_argument("--cascade", action="store_true",
                        help="apply the two models as a cascade: noise vs neural, then SUA vs MUA on the neural units")
    parser.add_argument("--output", default="unitrefine_predictions", help="folder to save the predictions in")
    parser.add_argument("--models_folder", default=None,
                        help=f"where models from the Hugging Face Hub are kept (default: ${MODELS_FOLDER_VARIABLE} "
                             f"or {DEFAULT_MODELS_FOLDER})")
    parser.add_argument("--task_index", type=int, default=None,
                        help="only label this analyzer, counting from 0, e.g. $SLURM_ARRAY_TASK_ID")
    parser.add_argument("--compute_missing_metrics", action="store_true",
                        help="compute the metrics the models need which an analyzer doesn't have, and save them in "
                             "the analyzer (needs its recording)")
    parser.add_argument("--n_jobs", type=int, default=1,
                        help="processes used to compute metrics, e.g. $SLURM_CPUS_PER_TASK")
    parser.add_argument("--max_depth", type=int, default=3, help="how many folder levels down to look for analyzers")
    parser.add_argument("--list", action="store_true", help="only list the analyzers and their numbers")
    args = parser.parse_args(argv)

    try:
        if args.list:
            analyzer_paths = collect_analyzers(args.analyzers, args.max_depth)
            for index, path in enumerate(analyzer_paths):
                print(f"{index}\t{path}")
            print(f"{len(analyzer_paths)} analyzers. For a SLURM job array: --array=0-{len(analyzer_paths) - 1}")
            return 0
        if len(args.models) == 0:
            parser.error("give at least one --model")
        results = apply_models(args.analyzers, args.models, cascade=args.cascade, output_folder=args.output,
                               models_folder=args.models_folder, task_index=args.task_index, max_depth=args.max_depth,
                               compute_missing_metrics=args.compute_missing_metrics, n_jobs=args.n_jobs)
    except UsageError as e:
        print(f"unitrefine-apply: {e}", file=sys.stderr)
        return 2

    failed = [result for result in results if "error" in result]
    print(f"\nLabelled {len(results) - len(failed)}/{len(results)} analyzers. Predictions saved in {Path(args.output).resolve()}.")
    for result in failed:
        print(f"Not labelled: {result['path']}: {result['error']}", file=sys.stderr)
    if any("required metrics" in result["error"] for result in failed) and not args.compute_missing_metrics:
        print("To compute the missing metrics and save them in the analyzers, add --compute_missing_metrics.", file=sys.stderr)
    return 1 if len(failed) > 0 else 0


def download_main(argv=None):
    parser = argparse.ArgumentParser(
        prog="unitrefine-download-models",
        description="Downloads UnitRefine models from the Hugging Face Hub, e.g. before running unitrefine-apply on "
                    "compute nodes without internet. Without repo ids, lists the published UnitRefine models.",
    )
    parser.add_argument("repo_ids", nargs="*", help="Hugging Face repo ids, e.g. SpikeInterface/UnitRefine_sua_mua_classifier")
    parser.add_argument("--models_folder", default=None,
                        help=f"where to save them (default: ${MODELS_FOLDER_VARIABLE} or {DEFAULT_MODELS_FOLDER})")
    args = parser.parse_args(argv)

    if len(args.repo_ids) == 0:
        print("UnitRefine models on the Hugging Face Hub:")
        for group in hfh_model_groups:
            for repo_id, note in group["models"]:
                print(f"  {repo_id}  {note}".rstrip())
        print("\nDownload them with: unitrefine-download-models REPO_ID [REPO_ID ...]")
        return 0

    status = 0
    for repo_id in args.repo_ids:
        try:
            model_folder = resolve_model(repo_id, args.models_folder)
            print(f"{repo_id}: {model_folder}")
        except UsageError as e:
            print(f"unitrefine-download-models: {e}", file=sys.stderr)
            status = 2
    return status


if __name__ == "__main__":
    sys.exit(apply_main())
