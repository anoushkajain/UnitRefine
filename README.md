# UnitRefine: A Community Toolbox for Automated Spike Sorting Curation  

**UnitRefine** is a machine-learning toolbox designed to streamline spike sorting curation by reducing the need for manual intervention. 
It integrates seamlessly with [SpikeInterface](https://github.com/SpikeInterface/spikeinterface) and supports both pre-trained models and custom model training.
UnitRefine is agnostic to probe type, species, brain region, and spike sorter, and includes a user-friendly GUI (using [SpikeInterface-GUI](https://github.com/SpikeInterface/spikeinterface-gui/) as a backend) for curation, training, validation, and retraining. The GUI also supports active learning, allowing users to iteratively improve model performance through targeted relabeling.

## Available Pre-trained Models

UnitRefine provides [pre-trained models](https://huggingface.co/AnoushkaJain3) for Single-Unit-Activity (SUA) identification across multiple datasets, probe types, and species:

| Dataset                      | Probe type                | n recordings | Spike sorter          | Species |
|------------------------------|----------------------------|--------------|------------------------|---------|
| Base dataset                 | Neuropixels 1.0            | 11           | Kilosort 2.5          | Mouse   |
| rat recordings               | Neuropixels 2.0            | 4            | Kilosort 4 (Pachitariu et al. 2024)          | rat   |
| Mole rat recordings          | Neuropixels 2.0            | 4            | Kilosort 4           | Mole rat (Shirdhankar et al. 2025) |
| Nonhuman primate recordings  | Utah array                 | 11           | Kilosort 4 | Macaque (Xing Chen et al. 2022)|
| Human intracranial recordings| Behnke–Fried electrodes    | 12           | Combinato (Niediek et al., 2016) | Human |
---
## Installation
To use UnitRefine, install SpikeInterface (≥ 0.102).
```bash
pip install spikeinterface
```
We provide a **UnitRefine GUI** that simplifies unit curation, model training, loading, and relabeling.

<p align="center">
  <img src="https://github.com/anoushkajain/UnitRefine/blob/main/src/unitrefine/resources/unitrefine_gui.JPG" width="500"/>
</p>

For detailed instructions and usage examples, please refer to the documentation [here](https://github.com/anoushkajain/UnitRefine/blob/main/src/unitrefine/README.md).

---
## Tutorials  
Also refer to the automated curation tutorials available in the SpikeInterface documentation:  
[Automated Curation Tutorials](https://spikeinterface.readthedocs.io/en/latest/tutorials_custom_index.html#automated-curation-tutorials)  

Additionally, this repository includes **Jupyter Notebooks** in [section](https://github.com/anoushkajain/UnitRefine/tree/main/UnitRefine/tutorial) with detailed step-by-step tutorials on how to:  
1. Apply pre-trained models.  
2. Train your own classifiers.   

---
## Citation

If you find **UnitRefine** useful in your research, please cite our preprint: https://www.biorxiv.org/content/10.1101/2025.03.30.645770v1

## Acknowledgements

We would like to express my sincere gratitude to the following individuals for their invaluable contributions to this project:
UnitRefine is highly dependent on the flexible and powerful SpikeInterface and Spikeinterface-GUI packages. Many thanks to Alessio, Sam, Zack, Joe who gave help and feedback to this project, and to the entire SpikeInterface team.

- **Code Refactoring and Integration in SpikeInterface:**  
  Chris Halcrow, Jake Swann, Robyn Greene, Sangeetha Nandakumar (IBOTS)

- **Model Curators:**  
  Nilufar Lahiji, Sacha Abou Rachid, Severin Graff, Luca Koenig, Natalia Babushkina, Simon Musall  

- **Advisors and collaborators:**  
  Alessio Buccino, Olivier Winter, Sonja Grün, Matthias Hennig, Simon Musall  


## Feedback and Contributions  
We encourage feedback, contributions, and collaboration from the community to improve UnitRefine. Feel free to open issues or submit pull requests to enhance the toolbox further.  


