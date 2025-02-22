# UnitRefine: A Community Toolbox for Automated Spike Sorting Curation  

**UnitRefine** is a machine-learning-based toolbox designed to streamline spike sorting curation by reducing the need for manual intervention.  

With a focus on accessibility and broad community adoption, UnitRefine offers:  
- Seamless integration with **SpikeInterface**.  
- Pre-trained machine learning models for effortless application.  
- The flexibility to train custom models using your own curated datasets and metrics.  
- Easy sharing of trained models via **Hugging Face Hub**, fostering collaboration and reproducibility.
- A manually curated dataset, labeled by 7 experts across 11 Neuropixels 1.0 recordings in mice, is also available.
- Each recording was annotated by 2 to 5 people, with an agreement rate of 80% among the curators.

## Key Features  
1. **Pre-trained Models**: Apply ready-to-use classifiers for noise removal and unit refinement.  
2. **Custom Training**: Train models on your own data to meet specific experimental needs.  
3. **Integration**: Fully integrated with SpikeInterface for a smooth user experience.  
4. **Models**: Share or download models from the [Hugging Face Hub](https://huggingface.co/collections/SpikeInterface/curation-models-66ebf53d7f17088991062c2c), enabling community-driven advancements.
5. **Agnostic** to probe type, species, brain region & spike sorter.

## Citation
If you find **UnitRefine** useful in your research, please cite the following DOI: [https://doi.org/10.6084/m9.figshare.28282841.v2](https://doi.org/10.6084/m9.figshare.28282841.v2).
We will be releasing a pre-print soon. In the meantime, please use the above DOI for referencing.

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


