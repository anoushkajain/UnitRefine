## Models for Unit Classification

Pre-trained machine learning models to streamline the classification of neural activity.
These models are designed to integrate seamlessly with the SpikeInterface pipeline, making it easier to curate and analyze large-scale electrophysiological datasets.

Some of the most relevant models are:
- **`noise_neural_classifier`**: Distinguishes between noise and neural signals.
- **`sua_mua_classifier`**: Differentiates between single-unit activity (SUA) and multi-unit activity (MUA).
- **`UnitRefine-mice-sua-classifier`**: Model trained on a large set of cluster features from mouse Neuropixels recordings in many brain regions. Differentiates between SUA and non-SUA clusters. Can be used as a starting model for a mouse-Neuropixels dataset.
- **`UnitRefine-mice-sua-classifier`**: Model trained on a large set of cluster features from mouse Neuropixels recordings in many brain regions. Differentiates between SUA and non-SUA clusters. Can be used as a starting model for a mouse-Neuropixels dataset. Specialized models for other species are also available in the collection.
- **`UnitRefine-generalized-sua-classifier`**: Model trained on a light-weight subset of cluster features from a diverse collection of data from different species and recording modalities. Differentiates between SUA and non-SUA clusters. Can be used as a starting model for a custom dataset for which no specialized model exists.

You can access and download these models from the [our Curation Models Collection on Hugging Face Hub](https://huggingface.co/AnoushkaJain3). Additional models from the SpikeInterface collection are also available [here](https://huggingface.co/collections/SpikeInterface/curation-models-66ebf53d7f17088991062c2c).
