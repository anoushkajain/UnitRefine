## Tutorial: Using the UnitRefine GUI

## Installation

1. To use our GUI, [Install uv](https://docs.astral.sh/uv/getting-started/installation/), the modern python package manager.
(Note for Windows users: If you have issues installing uv, please check out the FAQ section.)

2. Clone UnitRefine repository and move into the repo folder.

```bash
git clone https://github.com/anoushkajain/UnitRefine.git
cd UnitRefine
``` 
  
---
## Launching the GUI

Open UnitRefine, creating a new project.

```bash
uv run unitrefine --project_folder my_new_project
``` 
(Note: you must be in the UnitRefine folder that you've cloned from github when you run this command.)

A window should pop up that looks something like this:

<p align="center">
  <img src="https://github.com/anoushkajain/UnitRefine/blob/main/src/unitrefine/resources/unitrefine_gui.JPG" width="500"/>
</p>


### Trying the GUI

To try the GUI, you need [Sorting Analyzer](https://spikeinterface.readthedocs.io/en/stable/tutorials/core/plot_4_sorting_analyzer.html),
download a precomputed Sorting Analyzer from [here](https://drive.google.com/drive/folders/14axIOdweMeSpxigYlRIph4e7ZhO7_inN?usp=sharing)
and load it directly into the GUI (recommended).
Alternatively, you can load an example Allen Institute dataset by selecting Add Analyzer from S3 and pasting:

```bash
s3://aind-open-data/ecephys_820459_2025-11-10_15-07-13_sorted_2025-11-22_08-46-30/postprocessed/experiment1_Record Node 101#Neuropix-PXI-100.ProbeA-AP_recording1.zarr
```

Once you see the message **“Successfully added analyzer”** in your terminal, the dataset has been loaded correctly.

Click the **Curate** button as shown below to open the SpikeInterface GUI.
<p align="center">
  <img src="https://github.com/anoushkajain/UnitRefine/blob/main/src/unitrefine/resources/curation_gui.jpg" width="500"/>
</p>
**Note:** The SpikeInterface GUI may take some time to launch. While it is open, the UnitRefine GUI may appear unresponsive in the current version.

During curation, relabel units using the keyboard shortcuts:
- **`n`** → noise  
- **`g`** → good (SUA)  
- **`m`** → MUA
 
Once you have finished labeling the units, close the SpikeInterface GUI.  
Your labels are automatically saved in the project folder.

Then, click the Train button to train a model using your curated dataset.
<p align="center">
  <img src="https://github.com/anoushkajain/UnitRefine/blob/main/src/unitrefine/resources/train_and_load_gui.JPG" width="500"/>
</p>

In the terminal, you will see logs showing how your labels were used to train the model, for example:
```bash
Running RandomForestClassifier with imputation knn and scaling StandardScaler()
    Balanced Accuray: 
    Precision: 
    Recall: 
```
If your balanced accuracy is above 75%, your labeling is generally consistent and reliable. You can inspect the model output looking at inspect button.
If performance is lower, we recommend relabeling the units and retraining the model to improve results.

<p align="center">
  <img src="https://github.com/anoushkajain/UnitRefine/blob/main/src/unitrefine/resources/re-label.JPG" width="500"/>
</p>


## Loading Your Model

Once you have a trained model, you can use it to make predictions on new recordings.  
For each new recording, you first need to upload the corresponding sorting analyzer.

Then you can then either load a model from your local drive or use a pretrained model from the Hugging Face Hub (HFH).
for example:

```bash
SpikeInterface/UnitRefine_sua_mua_classifier
```
<p align="center">
  <img src="https://github.com/anoushkajain/UnitRefine/blob/main/src/unitrefine/resources/predict-gui.JPG" width="500"/>
</p>
The model predictions are saved in the project folder under:  
`analyzer_folder/labels_from_UnitRefine_sua_mua_classifier.csv`

You can inspect the predicted labels by clicking the Inspect button in the GUI.

If you think the model needs further improvement, you can relabel the units and retrain the model.


---

Now, you can know how to add sorting analyzers, curate the data, train a model, and validate its performance.  
Keep an eye on the feedback printed in the terminal—it provides helpful guidance throughout the process.

You can also generate Python code from the GUI and reuse it later in your own scripts.  

The next time you run **UnitRefine**, simply point to the same project folder and everything will be loaded automatically.


```bash
uv run unitrefine --project_folder my_existing_project
```
---
## FAQs

1. Issues with installing UV 
For windows users trying to install uv, try doing 
```bash
pip install uv
```
if this doesn't work then type the following on your cmd. 
```bash
powershell -ExecutionPolicy ByPass -c "irm https://astral.sh/uv/install.ps1 | iex"
```
