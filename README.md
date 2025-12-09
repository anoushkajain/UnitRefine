# UnitRefine: A Community Toolbox for Automated Spike Sorting Curation  

**UnitRefine** is a machine-learning toolbox designed to streamline spike sorting curation by reducing the need for manual intervention.
It integrates seamlessly with SpikeInterface and supports both pre-trained models and custom model training.
UnitRefine is agnostic to probe type, species, brain region, and spike sorter, and includes a user-friendly GUI for curation, training, validation, and retraining. The GUI also supports active learning, allowing users to iteratively improve model performance through targeted relabeling

## Available Pre-trained Models

UnitRefine provides [pre-trained models](https://huggingface.co/AnoushkaJain3)for Single-Unit-Activity (SUA) identification across multiple datasets, probe types, and species:

| Dataset                      | Probe type                | n recordings | Spike sorter          | Species |
|------------------------------|----------------------------|--------------|------------------------|---------|
| Base dataset                 | Neuropixels 1.0            | 11           | Kilosort 2.5          | Mouse   |
| rat recordings               | Neuropixels 2.0            | 4            | Kilosort 4            | rat   |
| Mole rat recordings          | Neuropixels 2.0            | 4            | Kilosort 4            | Mole rat |
| Nonhuman primate recordings  | Utah array                 | 11           | Kilosort 4 (Xing Chen et al. 2022)| Macaque |
| Human intracranial recordings| Behnke–Fried electrodes    | 12           | Combinato (Niediek et al., 2016) | Human |

---
## Citation

If you find **UnitRefine** useful in your research, please cite our preprint: https://www.biorxiv.org/content/10.1101/2025.03.30.645770v1


## Installation

Simply install Spikeinterface to use UnitRefine in your existing workflows 

```bash
pip install spikeinterface[full]
```

To use our GUI, [Install uv](https://docs.astral.sh/uv/getting-started/installation/), the modern python package manager.

1. Clone this repository and move into the repo folder

```bash
git clone https://github.com/anoushkajain/UnitRefine.git
cd UnitRefine
```

2. Open unit_refine, creating a new project

```bash
uv run unit_refine --project_folder my_new_project
``` 
Note: you must be in the UnitRefine folder that you've cloned from github when you run this command.
   
A window should pop up that looks something like this:






### Tutorials  
To get started with UnitRefine, refer to the automated curation tutorials available in the SpikeInterface documentation:  
[Automated Curation Tutorials](https://spikeinterface.readthedocs.io/en/latest/tutorials_custom_index.html#automated-curation-tutorials)  

Additionally, this repository includes **Jupyter Notebooks** in [section](https://github.com/anoushkajain/UnitRefine/tree/main/UnitRefine/tutorial) with detailed step-by-step tutorials on how to:  
1. Apply pre-trained models.  
2. Train your own classifiers.   


## Acknowledgements

I would like to express my sincere gratitude to the following individuals for their invaluable contributions to this project:

- **Code Refactoring and Integration in SpikeInterface:**  
  Chris Halcrow, Jake Swann, Robyn Greene, Sangeetha Nandakumar (IBOTS)

- **Model Curators:**  
  Nilufar Lahiji, Sacha Abou Rachid, Severin Graff, Luca Koenig, Natalia Babushkina, Simon Musall  

- **Advisors and collaborators:**  
  Alessio Buccino, Sonja Grün, Matthias Hennig, Simon Musall  


## Feedback and Contributions  
We encourage feedback, contributions, and collaboration from the community to improve UnitRefine. Feel free to open issues or submit pull requests to enhance the toolbox further.  


