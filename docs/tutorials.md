# Tutorials

## The GUI

The [GUI tutorial](https://github.com/anoushkajain/UnitRefine/blob/main/src/unitrefine/README.md) shows how to add
recordings, label units, train a model, improve it with active learning, and apply it to new recordings.

## Notebooks

The [Jupyter notebooks](https://github.com/anoushkajain/UnitRefine/tree/main/UnitRefine/tutorial) show how to:

1. **Apply pretrained models** to automatically curate spike sorted data
2. **Train your own classifiers** on manually curated labels
3. Use metrics computed beforehand and saved as `.csv` files
4. Use UnitRefine directly with SpikeInterface `SortingAnalyzer` objects

UnitRefine supports two workflows:

- **With a Sorting Analyzer (recommended).** Uses SpikeInterface `SortingAnalyzer` objects to compute the metrics,
  so they are the same as in SpikeInterface pipelines.
- **With a `.csv` of metrics.** Uses metrics computed beforehand, e.g. in your own pipeline outside SpikeInterface.

For more on automated curation in SpikeInterface, see the
[SpikeInterface automated curation tutorials](https://spikeinterface.readthedocs.io/en/latest/tutorials_custom_index.html#automated-curation-tutorials).

A typical workflow:

1. Run spike sorting (e.g. Kilosort) and compute metrics with SpikeInterface.
2. Apply a pretrained UnitRefine model, **or** label some units by hand.
3. Train or fine-tune a classifier.
4. Label all units automatically.
5. Optionally, improve the model with active learning.

## Applying models on a computing cluster

[`unitrefine-apply`](hpc.md) applies models to many recordings without the GUI, e.g. in a SLURM job array.

## Model interpretability with SHAP

[`SHAP_plots.ipynb`](https://github.com/anoushkajain/UnitRefine/blob/main/UnitRefine/plots/SHAP_plots.ipynb) shows,
as in the UnitRefine paper, how to compute SHAP values, check how stable the feature importance is across random
seeds, select the best model, and make confusion matrices.

## Questions

**How many units should I label?** At least 6 for each class you use (SUA, MUA, noise), or training fails. You can
also use only SUA and noise labels to train a binary instead of a 3-class classifier. For a first model with decent
performance, label at least 10% of the units, and more than 50 in total.
