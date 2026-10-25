# Installing UnitRefine

UnitRefine needs Python 3.11, 3.12 or 3.13, and runs on Linux, macOS and Windows on a standard computer.

## With pip (recommended)

We recommend installing UnitRefine in its own environment, e.g. with
[conda](https://docs.conda.io/projects/conda/en/latest/user-guide/install/index.html) and
[uv](https://docs.astral.sh/uv/) (a fast installer):

```bash
conda create -n unitrefine python=3.13
conda activate unitrefine
pip install uv
uv pip install "unitrefine[gui]"
```

Then start the GUI:

```bash
unitrefine --project_folder my_new_project
```

`unitrefine[gui]` installs UnitRefine with its GUI. To only apply models, e.g. on a computing cluster without a
screen, install it without the GUI: `uv pip install unitrefine` (see [Applying models on a computing cluster](hpc.md)).

## From the repository

To run the tutorial notebooks, or to change UnitRefine, clone the repository and install it with
[uv](https://docs.astral.sh/uv/getting-started/installation/):

```bash
git clone https://github.com/anoushkajain/UnitRefine.git
cd UnitRefine
uv sync
```

`uv sync` installs UnitRefine with its GUI, and the tools to run the tests (`uv run pytest`). Without the GUI:
`uv sync --no-group gui`. Then run the commands with `uv run` in front, from the repository's folder, e.g.
`uv run unitrefine --project_folder my_new_project`.

## With a conda environment

Recommended on Windows 11 with Smart App Control enabled (see below). An `environment.yml` is provided in the
repository:

```bash
conda env create -f environment.yml
conda activate unitrefine
pip install -e . --no-deps
```

The `--no-deps` is required so pip does not overwrite the conda-forge Qt binaries.

## What gets installed

- NumPy, Pandas, scikit-learn, SpikeInterface
- Hugging Face Hub (model loading) and skops (model files)
- With the GUI (`unitrefine[gui]`): spikeinterface-gui, PyQt5, pyqtgraph, matplotlib, SHAP

## Problems

**Installing uv fails on Windows.** Try `pip install uv`. If this does not work, follow the instructions on the
uv [Windows installation page](https://docs.astral.sh/uv/getting-started/installation/#__tabbed_1_2).

**"Application Control policy has blocked this file" on Windows 11.** Windows 11 Smart App Control blocks the
unsigned `.pyd` files that ship in PyPI wheels (e.g. `numcodecs/blosc.pyd`), which pip and uv install. Use the
conda environment above instead, or disable Smart App Control in Windows Security.
