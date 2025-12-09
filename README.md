# UnitRefine: A Community Toolbox for Automated Spike Sorting Curation  

**UnitRefine** is a machine-learning-based toolbox designed to streamline spike sorting curation by reducing the need for manual intervention. 
It integrates seamlessly with **SpikeInterface** and supports both **pre-trained models** and **custom model training**.

## Available Pre-trained Models

UnitRefine provides [pre-trained models](https://huggingface.co/AnoushkaJain3) across multiple datasets, probe types, and species:

| Dataset                      | Probe type                | n recordings | Spike sorter          | Species |
|------------------------------|----------------------------|--------------|------------------------|---------|
| Base dataset                 | Neuropixels 1.0            | 11           | Kilosort 2.5          | Mouse   |
| rat recordings               | Neuropixels 2.0            | 4            | Kilosort 4            | Mouse   |
| Mole rat recordings          | Neuropixels 2.0            | 4            | Kilosort 4            | Mole rat |
| Nonhuman primate recordings  | Utah array                 | 11           | Kilosort 4            | Macaque |
| Human intracranial recordings| Behnke–Fried electrodes    | 12           | Combinato (Niediek et al., 2016) | Human |


## Key Features

- **Pre-trained models** for Single-Unit-Activity (SUA) identificaation. 
- **Custom model training** using your own curated data  
- **Full integration** with the SpikeInterface ecosystem  
- **Model sharing** via the [Hugging Face Hub](https://huggingface.co/collections/SpikeInterface/curation-models-66ebf53d7f17088991062c2c)  
- **Agnostic** to probe type, species, brain region, and spike sorter  
- **User-friendly GUI** for training, validation, and curation  

---
## Citation

If you find **UnitRefine** useful in your research, please cite our preprint: https://www.biorxiv.org/content/10.1101/2025.03.30.645770v1


## Installation

To use this package, you can install it in two ways:

### 1. Install with `pyproject.toml` to use UnitRefine 

If you want to use **UnitRefine**, install the package using:

```bash
pip install .
```
### 2. Simply install Spikeinterface to use UnitRefine in your existing workflows 

```bash
pip install spikeinterface[full]
```

More installation instructions can be found [here](https://spikeinterface.readthedocs.io/en/latest/get_started/installation.html).  

### Tutorials  
To get started with UnitRefine, refer to the automated curation tutorials available in the SpikeInterface documentation:  
[Automated Curation Tutorials](https://spikeinterface.readthedocs.io/en/latest/tutorials_custom_index.html#automated-curation-tutorials)  

Additionally, this repository includes **Jupyter Notebooks** in [section](https://github.com/anoushkajain/UnitRefine/tree/main/UnitRefine/tutorial) with detailed step-by-step tutorials on how to:  
1. Apply pre-trained models.  
2. Train your own classifiers.   

## Reference Scripts

This repository contains two scripts, `model_based_curation.py` and `train_manual_curation.py`, that provide a detailed explanation of how certain features work when integrated with the [SpikeInterface](https://github.com/SpikeInterface) library. 

### Important Notes:
- These scripts **cannot be used independently**. They are designed for understanding the inner workings of SpikeInterface-related functionalities.
- For seamless integration and practical use, please install and use the official [SpikeInterface repository](https://github.com/SpikeInterface).
- These scripts rely on features already available in the SpikeInterface library.

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


