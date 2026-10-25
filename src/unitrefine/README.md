# UnitRefine GUI tutorial

This tutorial goes through the UnitRefine window section by section: label your units with the pretrained models,
train your own model, improve it by relabelling the units it is least sure about, and share it.

- [Install and open UnitRefine](#install-and-open-unitrefine)
- [Example data](#example-data)
- [1. Project](#1-project)
- [2. Label your units and train a model](#2-label-your-units-and-train-a-model)
- [3. Apply models](#3-apply-models)
- [4. Improve the model: relabel and retrain](#4-improve-the-model-relabel-and-retrain)
- [5. Share your model](#5-share-your-model)
- [Questions](#questions)

## Install and open UnitRefine

Install UnitRefine with its GUI (other ways are described in
[Installation](https://github.com/anoushkajain/UnitRefine/blob/main/docs/installation.md)):

```bash
conda create -n unitrefine python=3.13
conda activate unitrefine
pip install uv
uv pip install "unitrefine[gui]"
```

Open UnitRefine with a project folder:

```bash
unitrefine --project_folder my_project
```

A new folder becomes a new project. The next time, give the same folder and everything you did is opened again.
A new project downloads three pretrained models from the Hugging Face Hub (about 30 MB). To start without them,
add `--no_pretrained_models`.

<p align="center">
  <img src="https://raw.githubusercontent.com/anoushkajain/UnitRefine/main/docs/images/gui_window.png" width="700"/>
</p>

The window has five sections. Click a section's title to open or fold it. A new project opens with section 3, where
most people start. Keep an eye on the terminal: UnitRefine prints what it is doing there.

## Example data

UnitRefine works on [Sorting Analyzers](https://spikeinterface.readthedocs.io/en/stable/tutorials/core/plot_4_sorting_analyzer.html)
from SpikeInterface, or on Kilosort output curated in Phy (UnitRefine makes the analyzer for you).

To try it, use **+ Analyzer from s3** and paste this public recording from the Allen Institute:

```
s3://aind-open-data/ecephys_820459_2025-11-10_15-07-13_sorted_2025-11-22_08-46-30/postprocessed/experiment1_Record Node 101#Neuropix-PXI-100.ProbeA-AP_recording1.zarr
```

Or download a [precomputed Sorting Analyzer](https://drive.google.com/file/d/1TynO9qXTXm_IKRGtuOb8rCd-S0Zhza0I/view?usp=sharing),
unzip it, and add the folder `sorting_analyzer_folder` with **+ Sorting Analyzer folder**.

If you only have a table of metrics, not an analyzer, see the
[tutorial notebooks](https://github.com/anoushkajain/UnitRefine/blob/main/docs/tutorials.md).

## 1. Project

Shows the project's folders (with **Open** buttons), the labels used (good, MUA, noise) and how many analyzers,
labels and models the project has. Everything UnitRefine makes is saved in the project folder:

- `analyzers/`: your labels and the models' predictions for each analyzer, as `.csv` files
- `models/`: the models you trained or downloaded
- `figures/`: the figures shown in each section

## 2. Label your units and train a model

<p align="center">
  <img src="https://raw.githubusercontent.com/anoushkajain/UnitRefine/main/docs/images/gui_section2_label_and_train.png" width="700"/>
</p>

**Add your recordings.** Use **+ Sorting Analyzer folder**, **+ Analyzer from s3** or **+ Phy output**. Each one
appears as a row.

**Label their units.** Either:

- **Curate** opens the recording in the [SpikeInterface-GUI](https://github.com/SpikeInterface/spikeinterface-gui/).
  Label units with the keys **g** (good, SUA), **m** (MUA) and **n** (noise), and move between units with
  **Ctrl + up/down**. Close the SpikeInterface-GUI when you are done: your labels are saved in the project.
  Opening a recording, especially from s3, can take a while, and the UnitRefine window waits until you close the
  SpikeInterface-GUI.
- **+ Import labels** reads labels you already have: Phy (`cluster_group.tsv`, `cluster_info.tsv`), SortingView
  (`.json`), SpikeInterface curation (`.json`), or a `.csv`/`.tsv` with `unit_id` and `quality` columns. Labels good,
  MUA and noise are used; others (e.g. unsorted) are skipped.
- **+ Phy output** adds a Kilosort output folder curated in Phy. UnitRefine makes an analyzer from Phy's units and
  the recording Kilosort sorted (this takes a few minutes), saves it next to the Phy folder, and imports the labels
  you set in Phy. If the Phy folder was made from a SpikeInterface analyzer (with `export_to_phy`), no new analyzer
  is made: Phy's merges and splits are applied to that analyzer.

<p align="center">
  <img src="https://raw.githubusercontent.com/anoushkajain/UnitRefine/main/src/unitrefine/resources/si-gui.JPG" width="500"/>
</p>

Label at least about 10% of the units, and more than 50 in total, for a first model with decent performance.

**Train a model.** **Train a model** opens a window where you choose:

- **Which models to train:**
  - *SUA vs all*: good against everything else (MUA and noise become `not-sua`)
  - *Noise vs Neural* and *SUA vs MUA*: two models which together form a cascade, giving good, MUA or noise
- **Which metrics:** all available, or only the spike times, quality or template metrics
- **How to train them:** the classifier, scaler and imputer. The defaults (random forest, standard scaler,
  nearest-neighbour imputation) work well for most recordings and train quickly. If you tick more, each combination
  is trained and the best one is kept, which takes longer.
- **Test size:** the share of your labels held out to test the model (0.2 by default)

Click **Train models**. The models are saved in `models/` and appear in section 3. The terminal shows each model's
balanced accuracy on the held-out units: above about 75% means your labels are consistent. If it is lower, label more
units, or use section 4.

The figures below the list show your labels, the metrics of each label, and how each model does on the held-out
units. Click a figure to see it bigger.

## 3. Apply models

<p align="center">
  <img src="https://raw.githubusercontent.com/anoushkajain/UnitRefine/main/docs/images/gui_section3_apply.png" width="700"/>
</p>

**3.1 Load models.** Models you trained in section 2, and the pretrained models of a new project, are already here.

- **+ Load** adds a model folder from your computer, e.g. from another project.
- **+ Load from HFH** downloads a model from the Hugging Face Hub, e.g. `AnoushkaJain3/UnitRefine-mice-sua-classifier`.
  See [all UnitRefine models](https://huggingface.co/AnoushkaJain3).
- **Manage models** shows the labels each model was trained on, and deletes models.

The pretrained models of a new project:

| Model | Labels |
|-------|--------|
| [noise vs neural](https://huggingface.co/SpikeInterface/UnitRefine_noise_neural_classifier) + [SUA vs MUA](https://huggingface.co/SpikeInterface/UnitRefine_sua_mua_classifier), as a cascade | good (SUA), MUA, noise |
| [generalized SUA classifier](https://huggingface.co/AnoushkaJain3/UnitRefine-generalized-sua-classifier) | SUA, not SUA |

**3.2 Choose your models.** Use one model, or a **cascade** of a noise vs neural model and a SUA vs MUA model, which
labels each unit good, MUA or noise. A new project has the pretrained cascade chosen already. Tick several models to
compare them on the same recordings.

**3.3 Apply models to get predictions and inspect.** Add the recordings you want labels for with
**+ Sorting Analyzer folder** (one analyzer, or a folder with several), **+ Analyzer from s3** or **+ Phy output**.
If a recording doesn't have all the metrics the models use, UnitRefine offers to compute them.

- **Inspect** opens the recording in the SpikeInterface-GUI with each model's label and confidence for every unit.
  Sort by confidence to check the units the model is least sure about.
- **+ Import labels** (optional) adds your own labels for a recording, to compare them with the predictions.

The predictions are saved in the project, e.g. `analyzers/analyzer_0/labels_from_<model>.csv`. The figures show the
predicted labels, the confidence, and how the predictions compare with your labels.

When a model predicts `good` vs `not-sua`, your MUA and noise labels count as `not-sua` when they are compared.

**Generate code to apply the model(s) in a script** prints Python code to the terminal, to apply the chosen models
in your own pipeline. To apply models to many recordings, e.g. on a computing cluster, use
[`unitrefine-apply`](https://github.com/anoushkajain/UnitRefine/blob/main/docs/hpc.md).

## 4. Improve the model: relabel and retrain

<p align="center">
  <img src="https://raw.githubusercontent.com/anoushkajain/UnitRefine/main/docs/images/gui_section4_relabel.png" width="700"/>
</p>

Active learning: label the units the model is least sure about, which improves it fastest.

**4.1** Set the number of units to relabel, and click **Find least confident units**. They are picked across all
   your analyzers, from the units you have not labelled yet, using the models chosen in 3.2.
   
**4.2** Click **Relabel** on each analyzer to label its picked units in the SpikeInterface-GUI.

**4.3** Choose the model to retrain and click **Retrain**. The new model is saved as e.g.
   `model_1_retrained_01`, then `model_1_retrained_02`, and so on.
   
**4.4** The retraining history shows each round.

Repeat for a few rounds, and stop when the *Accuracy over rounds* figure levels off.

## 5. Share your model

Upload your model to the Hugging Face Hub so others can use it with **+ Load from HFH**: see the
[SpikeInterface tutorial on uploading a model](https://spikeinterface.readthedocs.io/en/stable/tutorials/curation/plot_3_upload_a_model.html).

## Questions

**How many units should I label?** At least 6 for each label you use (good, MUA, noise), or training fails. For a
first model with decent performance, label at least 10% of the units, and more than 50 in total. With only good and
noise labels, train *SUA vs all*.

**Installing fails.** See [Installation](https://github.com/anoushkajain/UnitRefine/blob/main/docs/installation.md#problems).

If you find other problems, please [open an issue](https://github.com/anoushkajain/UnitRefine/issues).
