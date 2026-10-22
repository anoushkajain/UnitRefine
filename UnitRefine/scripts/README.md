## How training and applying models works

UnitRefine's models are trained and applied with functions built into [SpikeInterface](https://github.com/SpikeInterface/spikeinterface).
To see how they work, read their source code there, which is always up to date with the latest SpikeInterface version:

- **Training a model**: [`train_manual_curation.py`](https://github.com/SpikeInterface/spikeinterface/blob/main/src/spikeinterface/curation/train_manual_curation.py), used through `spikeinterface.curation.train_model`.
- **Applying a model**: [`model_based_curation.py`](https://github.com/SpikeInterface/spikeinterface/blob/main/src/spikeinterface/curation/model_based_curation.py), used through `spikeinterface.curation.model_based_label_units` (called `auto_label_units` before SpikeInterface 0.104).
- **The UnitRefine cascade** (noise vs neural, then SUA vs MUA on the neural units, as in the paper): [`unitrefine_curation.py`](https://github.com/SpikeInterface/spikeinterface/blob/main/src/spikeinterface/curation/unitrefine_curation.py), used through `spikeinterface.curation.unitrefine_label_units`.

The tutorials in [`../tutorial`](../tutorial) show how to use these functions.

Earlier versions of this folder had copies of these files, which went out of date as SpikeInterface changed.
