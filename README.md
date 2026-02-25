# UnitRefine: A Community Toolbox for Automated Spike Sorting Curation  

**UnitRefine** is an open-source machine-learning framework for automated spike-sorting curation in electrophysiological experiments.  
It strongly reduces the need for manual curation of spike-sorting results by leveraging supervised classifiers trained on human-labeled data to predict **single-unit activity (SUA)**, **multi-unit activity (MUA)**, and **noise** in sorted clusters.

<p align="center">
  <img src="https://github.com/anoushkajain/UnitRefine/blob/main/src/unitrefine/resources/unit_types_example.jpg" width="500"/>
</p>

UnitRefine is fully integrated into [SpikeInterface](https://github.com/SpikeInterface/spikeinterface), enabling users to:

- Apply **pre-trained curation models** to new recordings  
- Train and fine-tune **custom models** on their own curated experimental data  
- Iteratively improve models using **active learning**  
- Share reproducible models via the Hugging Face Hub  

UnitRefine is agnostic to probe type, species, brain region, or spike sorter and generalizes to previously unseen datasets. It has been validated across high-density probes, Utah arrays, and intracranial human recordings from multiple laboratories and species.

A user-friendly GUI supports end-to-end workflows including curation (using [SpikeInterface-GUI](https://github.com/SpikeInterface/spikeinterface-gui/) for cluster visualization), training, validation, model loading, and retraining.  
The GUI also supports active learning by highlighting uncertain clusters, allowing users to iteratively improve model performance through targeted relabeling.

---
## Available Pre-trained Models
UnitRefine provides several [pre-trained models](https://huggingface.co/AnoushkaJain3) from different species and experimental setups.

Each model folder includes:

- The trained classifier (`.skops` format)  
- Model metadata  
- The curated feature matrix used for training  

In our [preprint](https://www.biorxiv.org/content/10.1101/2025.03.30.645770v2) we show that UnitRefine reliably identifies human-labeled Single-Unit Activity (SUA) across multiple datasets, probe types, and species.


| Dataset          | Species        | Probe type                 | Spike sorter                | Pipeline       | Output format           | Source |
|------------------|----------------|----------------------------|-----------------------------|----------------|-------------------------|--------|
| Base dataset     | Mouse          | Neuropixels 1.0            | Kilosort 2.5                | SpikeInterface | Kilosort folders        | UnitRefine base dataset |
| IBL dataset      | Mouse          | Neuropixels 1.0            | IBL sorter (PyKilosort 2.5) | IBL pipeline   | SortingAnalyzer objects | International Brain Laboratory     |
| Allen dataset    | Mouse          | Neuropixels 2.0            | Kilosort 4                  | Allen ecephys  | `.zarr` files           | Allen Institute |
| Mole rat dataset | Naked mole rat | Neuropixels 2.0            | Kilosort 4                  | SpikeInterface | SortingAnalyzer objects | [Shirdhankar et al., 2025](https://doi.org/10.64898/2025.12.15.693140) |
| Monkey dataset   | Rhesus macaque | Utah array                 | Kilosort 4                  | Custom         | Kilosort folders        | [Chen et al., 2022](https://www.nature.com/articles/s41597-022-01180-1) |
| Human dataset    | Human          | Behnke–Fried electrodes    | Combinato                   | Combinato      | Combinato output        | [Gerken et al., 2025](https://elifesciences.org/reviewed-preprints/106758) |


All datasets are also publicly available [on figshare](https://figshare.com/articles/dataset/Curated_dataset/28282799).

---
## Typical Workflow

1. Run spike sorting (e.g. Kilosort) and compute metrics with SpikeInterface.  
2. Apply a pre-trained UnitRefine model **or** label a subset of clusters manually.  
3. Train or fine-tune a classifier.  
4. Automatically curate the full dataset.  
5. Optionally refine using active learning.  

---

## Installation

1. [Install uv](https://docs.astral.sh/uv/getting-started/installation/) (recommended python package manager).  
(Note for Windows users: If you have issues installing uv, please check out the FAQ section.)

2. Use Git (https://git-scm.com/install) to clone the UnitRefine repository. Then move into the repo folder to install dependencies and run the GUI.

```bash
git clone https://github.com/anoushkajain/UnitRefine.git
cd UnitRefine
``` 
3. Install dependencies

```bash
uv sync
``` 
---

## Tutorials

We provide detailed **Jupyter Notebook tutorials** to help you get started with UnitRefine.

Tutorials are available in the repository under [`UnitRefine/tutorial`](https://github.com/anoushkajain/UnitRefine/tree/main/UnitRefine/tutorial)

The notebooks demonstrate how to:

1. **Apply pre-trained models** to automatically curate spike-sorted datasets  
2. **Train custom classifiers** using manually curated labels  
3. Use pre-computed cluster metrics stored as `.csv` files  
4. Integrate UnitRefine directly with SpikeInterface `SortingAnalyzer` objects  

UnitRefine supports two main workflows:

- **Analyzer-based workflow (recommended)**  
  Uses SpikeInterface `SortingAnalyzer` objects for metric computation and ensures consistency with SpikeInterface pipelines.

- **CSV-based workflow**  
  Uses pre-computed cluster metrics stored as `.csv` files.  
  This enables integration into custom pipelines outside of SpikeInterface.

For additional background on automated curation within the SpikeInterface ecosystem, see the official  
[SpikeInterface automated curation tutorials](https://spikeinterface.readthedocs.io/en/latest/tutorials_custom_index.html#automated-curation-tutorials).

---
## Model Interpretability with SHAP

For transparent and reproducible model interpretation (as described in the UnitRefine paper), we provide a dedicated notebook:
[`SHAP_plots.ipynb`](https://github.com/anoushkajain/UnitRefine/blob/main/UnitRefine/plots/SHAP_plots.ipynb)

This notebook demonstrates how to compute SHAP values, evaluate feature importance stability across random seeds, select the best-performing model, and generate confusion matrices for reproducible model interpretation.

---
## Launching the GUI
We provide a UnitRefine GUI that simplifies unit curation, model training, loading, and relabeling.  

For detailed instructions and usage examples, please refer to the GUI documentation [here](https://github.com/anoushkajain/UnitRefine/blob/main/src/unitrefine/README.md).

To run the GUI inside the UnitRefine repo, create a new project.

```bash
uv run unitrefine --project_folder my_new_project
``` 
> **Important:** This command must be executed from the root folder of the cloned UnitRefine repository.

This will create a new project folder and launch the UnitRefine GUI.
A window should pop up that looks something like this:

<p align="center">
  <img src="https://github.com/anoushkajain/UnitRefine/blob/main/src/unitrefine/resources/unitrefine_gui.JPG" width="500"/>
</p>

Within the GUI, users can:

- Visualize cluster waveforms, amplitudes, correlograms, and quality metrics  
- Manually assign or correct cluster labels  
- Train new models using curated labels  
- Load and apply pre-trained models  
- Validate model predictions  
- Retrain models based on updated labels  

The GUI also supports active learning by highlighting clusters with low prediction confidence, enabling efficient and targeted relabeling to improve model performance.

---

## System requirements

### Hardware requirements
UnitRefine requires only a standard computer with enough RAM to support the in-memory operations.

### Software requirements

**Tested on:** Linux • macOS • Windows  
**Python:** 3.11+

### Core dependencies (installed automatically)
- NumPy, Pandas  
- scikit-learn  
- SpikeInterface + spikeinterface-gui  
- PyQt5 (GUI backend)  
- Hugging Face Hub (model loading)  
- skops (model serialization)

## Citation
If you find **UnitRefine** useful in your research, please cite our preprint: https://www.biorxiv.org/content/10.1101/2025.03.30.645770v2

## License
This software is released under the MIT license.

## Acknowledgements

We would like to express our sincere gratitude to the following individuals for their invaluable contributions to this project:
UnitRefine builds heavily on the flexible and powerful SpikeInterface and SpikeInterface-GUI packages. Many thanks to Alessio, Sam, Zack, and Joe for their help and feedback on this project,
as well as to the entire SpikeInterface team.

- **Code Refactoring and Integration in SpikeInterface:**  
  Chris Halcrow, Jake Swann, Robyn Greene, Sangeetha Nandakumar (IBOTS)

- **Model Curators:**  
  Nilufar Lahiji, Sacha Abou Rachid, Severin Graff, Luca Koenig, Natalia Babushkina, Simon Musall  

- **Advisors and collaborators:**  
  Alessio Buccino, Olivier Winter, Sonja Grün, Matthias Hennig, Simon Musall  


## Feedback and Contributions  
We encourage feedback, contributions, and collaboration from the community to improve UnitRefine. Feel free to open issues or submit pull requests to enhance the toolbox further.  

---
## FAQs

**1. Issues with installing UV**

For Windows users trying to install uv, try doing 
```bash
pip install uv
```
If this does not work, please follow the instructions on the uv [Windows installation page.](https://docs.astral.sh/uv/getting-started/installation/#__tabbed_1_2)

 
**2. Number of labels**

You need to provide at least 6 labels for each used class (SUA, MUA, Noise) to prevent errors during model fitting (e.g. cross-validation and class balancing). You can also use only SUA and Noise labels to create a binary instead of a 3-class classifier. As a starting model with decent performance you should label at least 10% of the data (should be more than 50 clusters in total).
