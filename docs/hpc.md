# Applying models on a computing cluster (HPC)

UnitRefine's models can be applied without the GUI, e.g. to many recordings on a computing cluster. Two commands
do this. They don't need a screen and don't load Qt:

- `unitrefine-apply` applies models to Sorting Analyzers and saves their predictions, one csv per analyzer.
- `unitrefine-download-models` downloads models from the Hugging Face Hub, for compute nodes without internet.

The predictions are the same as the GUI's (section 3), with the same labels and columns, also for cascades.

## Install

On the cluster, UnitRefine can be installed without the GUI (Qt, pyqtgraph and the SpikeInterface-GUI), which isn't
needed to apply models:

```bash
git clone https://github.com/anoushkajain/UnitRefine.git
cd UnitRefine
uv sync --no-group gui
source .venv/bin/activate      # makes unitrefine-apply available
```

With pip, `pip install .` in the repository installs it without the GUI, and `pip install ".[gui]"` with it. A full
install (`uv sync`, as for the GUI) works on a cluster too.

## Get the models

A model is either a model folder, e.g. one trained in the GUI (`my_project/models/my_model`, copy it to the cluster),
or a Hugging Face repo id, which is downloaded the first time it is used. Compute nodes often have no internet, so
download the models first on a login node:

```bash
unitrefine-download-models          # lists the published UnitRefine models
unitrefine-download-models SpikeInterface/UnitRefine_noise_neural_classifier SpikeInterface/UnitRefine_sua_mua_classifier
```

They are saved in `~/.cache/unitrefine/models`. To keep them elsewhere, e.g. in a shared lab folder, give
`--models_folder FOLDER` or set `UNITREFINE_MODELS_FOLDER`, to both commands.

## Apply them

```bash
unitrefine-apply /data/sortings \
    --model SpikeInterface/UnitRefine_noise_neural_classifier \
    --model SpikeInterface/UnitRefine_sua_mua_classifier --cascade \
    --output /data/unitrefine_predictions
```

- **Analyzers:** give analyzer folders, folders with analyzers inside (searched 3 levels down, or `--max_depth N`),
  or text files with one analyzer path per line. `--list` shows the analyzers found, numbered from 0.
- **Models:** give `--model` once per model. Without `--cascade`, each model is applied on its own. With
  `--cascade`, give a noise vs neural model and a SUA vs MUA model, in either order: the SUA vs MUA model is applied
  to the units the first one calls neural.
- **Output:** one csv per analyzer, `{output}/{analyzer name}__labels_from_{models}.csv`, with a `unit_id` column,
  a `..._model_label` column per model (and `cascade_model_label` for a cascade), and their `..._confidence`.
  The analyzer names come from their folders, with parent folders added when names repeat, e.g.
  `mouse1_sorting_analyzer`.
- **Running it again** reuses the predictions which are newer than the models and the analyzer's metrics, so only
  new or changed analyzers are labelled.
- **Exit status:** 0 if every analyzer was labelled, 1 if some couldn't be (they are listed at the end), and 2 for
  a problem with the command, e.g. a model which doesn't exist.

### Missing metrics

A model can only be applied to an analyzer which has the metrics it was trained on. The models on the Hugging Face
Hub also use the drift metrics and the multi-channel template metrics (e.g. `spread`, `velocity_above`), which
analyzers often don't have. Add `--compute_missing_metrics` to compute the metrics the models need which an analyzer
doesn't have, and save them in the analyzer. The metrics it already has are kept.

- They are computed with SpikeInterface's default parameters, as the published models were trained with.
- This needs the analyzer's recording. An analyzer made on another computer looks for its recording at the path it
  had there, so compute the metrics where that path exists.
- `--n_jobs N` uses N processes, e.g. `--n_jobs $SLURM_CPUS_PER_TASK`.
- Computing the waveforms, which the PCA metrics need, makes SpikeInterface delete the templates and the
  extensions made from them. UnitRefine computes these again, with the parameters they had.

## A SLURM job array: one job per analyzer

List the analyzers once, and use the number of the last one in `--array`:

```bash
unitrefine-apply /data/sortings --list
```

Then submit this script with `sbatch apply_unitrefine.sh`:

```bash
#!/bin/bash
#SBATCH --job-name=unitrefine
#SBATCH --array=0-41               # one job per analyzer: 0 to (number of analyzers - 1)
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=01:00:00
#SBATCH --output=logs/unitrefine_%A_%a.out

source ~/UnitRefine/.venv/bin/activate
export HF_HUB_OFFLINE=1            # the models were downloaded on the login node

unitrefine-apply /data/sortings \
    --model SpikeInterface/UnitRefine_noise_neural_classifier \
    --model SpikeInterface/UnitRefine_sua_mua_classifier --cascade \
    --task_index "$SLURM_ARRAY_TASK_ID" \
    --compute_missing_metrics --n_jobs "$SLURM_CPUS_PER_TASK" \
    --output /data/unitrefine_predictions
```

Each job labels one analyzer. All jobs name their files from the whole list, so they don't overwrite each other.
Without `--compute_missing_metrics`, applying the models takes seconds per analyzer, so one job for all analyzers
is enough.

## From Python

```python
from unitrefine.cli import apply_models

results = apply_models(
    ["/data/sortings"],
    ["SpikeInterface/UnitRefine_noise_neural_classifier", "SpikeInterface/UnitRefine_sua_mua_classifier"],
    cascade=True,
    output_folder="/data/unitrefine_predictions",
)
# one dict per analyzer: its "path", and the predictions "file", or the "error" if it couldn't be labelled
```
