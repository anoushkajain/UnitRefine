# UnitRefine: a community toolbox for automated spike sorting curation

[![PyPI](https://img.shields.io/pypi/v/unitrefine)](https://pypi.org/project/unitrefine/)
[![Python](https://img.shields.io/pypi/pyversions/unitrefine)](https://pypi.org/project/unitrefine/)
[![Tests](https://github.com/anoushkajain/UnitRefine/actions/workflows/tests.yml/badge.svg)](https://github.com/anoushkajain/UnitRefine/actions/workflows/tests.yml)
[![License](https://img.shields.io/badge/license-MIT-blue)](https://github.com/anoushkajain/UnitRefine/blob/main/LICENSE)

**UnitRefine** labels spike sorted units as **single units (SUA)**, **multi-units (MUA)** or **noise** with machine
learning classifiers trained on human curation, so you don't have to curate every unit by hand. It works with any
probe, species, brain region and spike sorter, and is part of [SpikeInterface](https://github.com/SpikeInterface/spikeinterface).

<p align="center">
  <img src="https://raw.githubusercontent.com/anoushkajain/UnitRefine/main/src/unitrefine/resources/unit_types_example.jpg" width="500"/>
</p>

With UnitRefine you can:

- label your units with [pretrained models](#pretrained-models)
- train your own model on your labels, and improve it with active learning, in the [GUI](#the-gui)
- apply models to many recordings on a [computing cluster](https://github.com/anoushkajain/UnitRefine/blob/main/docs/hpc.md)
- share models on the [Hugging Face Hub](https://huggingface.co/AnoushkaJain3)

## Preprint

UnitRefine and its validation on mouse, naked mole rat, monkey and human recordings are described in our
[preprint](https://www.biorxiv.org/content/10.1101/2025.03.30.645770v2).

## The GUI

<p align="center">
  <img src="https://raw.githubusercontent.com/anoushkajain/UnitRefine/main/docs/images/gui_window.png" width="700"/>
</p>

```bash
unitrefine --project_folder my_project
```

A new project comes with the pretrained models. Add your spike sorted data in section 3.3 (a Sorting Analyzer, or a
Kilosort output folder curated in Phy) and UnitRefine model labels
its units. To train your own model, label units in the
[SpikeInterface-GUI](https://github.com/SpikeInterface/spikeinterface-gui/) or import labels you already have.

See the **[GUI tutorial](https://github.com/anoushkajain/UnitRefine/blob/main/src/unitrefine/README.md)** for all steps.

## Pretrained models

A new project downloads these three models from the Hugging Face Hub:

| Model | Labels |
|-------|--------|
| [noise vs neural](https://huggingface.co/SpikeInterface/UnitRefine_noise_neural_classifier) + [SUA vs MUA](https://huggingface.co/SpikeInterface/UnitRefine_sua_mua_classifier), used together as a cascade | good (SUA), MUA, noise |
| [generalized SUA classifier](https://huggingface.co/AnoushkaJain3/UnitRefine-generalized-sua-classifier) | SUA, not SUA |

UnitRefine was trained and tested on these datasets ([more models](https://huggingface.co/AnoushkaJain3)):

| Dataset | Species | Probe | Spike sorter |
|---------|---------|-------|--------------|
| UnitRefine base dataset | Mouse | Neuropixels 1.0 | Kilosort 2.5 |
| International Brain Laboratory (Witten Lab) | Mouse | Neuropixels 1.0 | IBL sorter (PyKilosort 2.5) |
| Allen Institute | Mouse | Neuropixels 2.0 | Kilosort 4 |
| [Shirdhankar et al., 2025](https://doi.org/10.64898/2025.12.15.693140) | Naked mole rat | Neuropixels 2.0 | Kilosort 4 |
| [Chen et al., 2022](https://www.nature.com/articles/s41597-022-01180-1) | Rhesus macaque | Utah array | Kilosort 4 |
| [Gerken et al., 2025](https://elifesciences.org/reviewed-preprints/106758) | Human | Behnke–Fried electrodes | Combinato |
The curated datasets are on [figshare](https://figshare.com/articles/dataset/Curated_dataset/28282799).

## Installation

```bash
conda create -n unitrefine python=3.13
conda activate unitrefine
pip install uv
uv pip install "unitrefine[gui]"
```

Without the GUI, e.g. on a computing cluster: `uv pip install unitrefine`. To install from the repository, with
conda only, or if installing fails, see [Installation](https://github.com/anoushkajain/UnitRefine/blob/main/docs/installation.md).

## Documentation

- [GUI tutorial](https://github.com/anoushkajain/UnitRefine/blob/main/src/unitrefine/README.md)
- [Tutorial notebooks](https://github.com/anoushkajain/UnitRefine/blob/main/docs/tutorials.md): apply and train models in Python, use a `.csv` of metrics, SHAP plots
- [Applying models on a computing cluster](https://github.com/anoushkajain/UnitRefine/blob/main/docs/hpc.md)
- [Installation](https://github.com/anoushkajain/UnitRefine/blob/main/docs/installation.md)
- [Automated curation in SpikeInterface](https://spikeinterface.readthedocs.io/en/latest/tutorials_custom_index.html#automated-curation-tutorials)

## Citation

If you find UnitRefine useful in your research, please cite:

```bibtex
@article{jain2025unitrefine,
  title={UnitRefine: A Community Toolbox for Automated Spike Sorting Curation},
  author={Jain, Anoushka and Greene, Robyn and Halcrow, Chris and Swann, Jake A and Kleinjohann, Alexander and Spurio, Federico and Graff, Severin and Pan-Vazquez, Alejandro and Kampa, Bj{\"o}rn and Gall, Juergen and Gr{\"u}n, Sonja and Winter, Olivier and Buccino, Alessio and Hennig, Matthias H and Musall, Simon},
  journal={bioRxiv},
  year={2025},
  doi={10.1101/2025.03.30.645770}
}
```

## Acknowledgements

UnitRefine builds on [SpikeInterface](https://github.com/SpikeInterface/spikeinterface) and
[SpikeInterface-GUI](https://github.com/SpikeInterface/spikeinterface-gui/). Many thanks to Alessio, Sam, Zack and Joe
for their help and feedback, and to the whole SpikeInterface team.

- **Code refactoring and integration in SpikeInterface:** Chris Halcrow, Jake Swann, Robyn Greene, Sangeetha Nandakumar (IBOTS)
- **Model curators:** Nilufar Lahiji, Sacha Abou Rachid, Severin Graff, Luca Koenig, Natalia Babushkina, Simon Musall
- **Advisors and collaborators:** Alessio Buccino, Olivier Winter, Sonja Grün, Matthias Hennig, Simon Musall

Feedback and contributions are welcome: please [open an issue](https://github.com/anoushkajain/UnitRefine/issues) or a
pull request. UnitRefine is released under the MIT license.
