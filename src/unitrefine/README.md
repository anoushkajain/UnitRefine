## Tutorial: Using the UnitRefine GUI

## Installation

1. To use our GUI, [Install uv](https://docs.astral.sh/uv/getting-started/installation/), the modern python package manager.

2. Clone UnitRefine repository and move into the repo folder.

```bash
git clone https://github.com/anoushkajain/UnitRefine.git
cd UnitRefine
``` 


### Launching the GUI

Open UnitRefine, creating a new project.

```bash
uv run unitrefine --project_folder my_new_project
``` 
Note: you must be in the UnitRefine folder that you've cloned from github when you run this command.

A window should pop up that looks something like this:

<p align="center">
  <img src="https://github.com/anoushkajain/UnitRefine/blob/main/src/unitrefine/resources/unitrefine_gui.JPG" width="500"/>
</p>


To try the GUI, you need [Sorting Analyzer](https://spikeinterface.readthedocs.io/en/stable/tutorials/core/plot_4_sorting_analyzer.html), you can load an example Allen Institute dataset by selecting **Add Analyzer from S3** and pasting:

```bash
s3://aind-open-data/ecephys_820459_2025-11-10_15-07-13_sorted_2025-11-22_08-46-30/postprocessed/experiment1_Record Node 101#Neuropix-PXI-100.ProbeA-AP_recording1.zarr
```
UnitRefine/src/unitrefine/resources/sorting_analyzer_folder

Once you see the message **“Successfully added analyzer”** in your terminal, the dataset has been loaded correctly.

Click the **Curate** button to open the SpikeInterface GUI.

> **Note:** The SpikeInterface GUI may take some time to launch. While it is open, the UnitRefine GUI may appear unresponsive in the current version.

During curation, relabel units using the keyboard shortcuts:
- **`n`** → noise  
- **`g`** → good (SUA)  
- **`m`** → MUA
 
















Then load a pretrained model from the Hugging Face Hub (HFH):
```bash
SpikeInterface/UnitRefine_sua_mua_classifier
```

From here, it should be easy to add sorting analyzers, curate the data, train a model and validate your model. Keep an eye on the feedback that comes through the terminal - it will help! You can also generate code which you could use in a Python script. Here is a flow chart to show the functionalities of the GUI

Whenever you curate something or make a model, whatever you've done is automatically saved in your project folder. Next time you run unit_refine, just point to your existing folder and it will load:
```bash
uv run unitrefine --project_folder my_existing_project
```

FAQs

1. Issues with installing UV 
For windows users trying to install uv, try doing 
```bash
pip install uv
```
if this doesn't work then type the following on your cmd. 
```bash
powershell -ExecutionPolicy ByPass -c "irm https://astral.sh/uv/install.ps1 | iex"
```
