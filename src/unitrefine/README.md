## Tutorial: Using the UnitRefine GUI

## Installation

1. To use our GUI, [Install uv](https://docs.astral.sh/uv/getting-started/installation/), the modern python package manager.<br>
(Note for Windows users: If you have issues installing uv, please check out the FAQ section.)

2. Use Git (https://git-scm.com/install) to clone the UnitRefine repository and move into the repo folder to launch the GUI.

```bash
git clone https://github.com/anoushkajain/UnitRefine.git
cd UnitRefine
``` 
  
---
## Launching the GUI

Open UnitRefine and create a new project.

```bash
uv run unitrefine --project_folder my_new_project
``` 
(Note: you must be in the UnitRefine folder that you've cloned from Github when you run this command.)

A window should pop up that looks something like this:

<p align="center">
  <img src="https://github.com/anoushkajain/UnitRefine/blob/main/src/unitrefine/resources/unitrefine_gui.JPG" width="500"/>
</p>


To use the GUI, you need a recording in form of a [Sorting Analyzer](https://spikeinterface.readthedocs.io/en/stable/tutorials/core/plot_4_sorting_analyzer.html) object.
If your data is saved in a different format and you want to train and apply a model based on an existing set of metrics, you can check out our tutorial notebooks [here](https://github.com/anoushkajain/UnitRefine/tree/main/UnitRefine/tutorial).

To test the GUI, you can also download a precomputed Sorting Analyzer from [here](https://drive.google.com/file/d/1TynO9qXTXm_IKRGtuOb8rCd-S0Zhza0I/view?usp=sharing). Just download the .zip file and uncompress it. Then load the contained folder "sorting_analyzer_folder" directly into the GUI using the "Load Analyzer folder" button (recommended). 

Alternatively, you can load an example Allen Institute dataset by selecting “Add Analyzer from S3” and pasting:

```bash
s3://aind-open-data/ecephys_820459_2025-11-10_15-07-13_sorted_2025-11-22_08-46-30/postprocessed/experiment1_Record Node 101#Neuropix-PXI-100.ProbeA-AP_recording1.zarr
```

Once you see the message **“Successfully added analyzer”** in your terminal, the dataset has been loaded correctly.

Now, click the **Curate** button as shown below to open the SpikeInterface GUI and curate the spike-sorted clusters.
<p align="center">
  <img src="https://github.com/anoushkajain/UnitRefine/blob/main/src/unitrefine/resources/curation_gui.jpg" width="500"/>
</p>
Note: It may take some time for data to download and the SpikeInterface GUI to launch. While it is open, the UnitRefine GUI may appear unresponsive in the current version. Just be patient until the data is loaded and remember to close the SpikeInterface GUI before returning to the UnitRefine GUI.
<p></p>

During curation in SpikeInterface GU, units can be relabeled using the keyboard shortcuts:

- **`n`** → noise  
- **`g`** → good (SUA)  
- **`m`** → MUA
 
You can also navigate between units using Ctr + up/down arrows.
Once you have finished labeling some units, close the SpikeInterface GUI. Your labels are automatically saved in the project folder.
To train a model, you should label at least around 10% of the clusters in the recording to achieve decent prediction accuracy.

## Train a model

Click the Train button to train a model using your curated dataset.
<p align="center">
  <img src="https://github.com/anoushkajain/UnitRefine/blob/main/src/unitrefine/resources/train_and_load_gui.JPG" width="500"/>
</p>

This will bring up a dialog to select the model type and pre-processing method. Our results from testing different models are described in the [UnitRefine paper](https://www.biorxiv.org/content/10.1101/2025.03.30.645770v2.full) but for most dataset, the default settings (random forest classifer with nearest-neighbor imputation and standard scaler) should work well and can be trained very quickly. Click "Train models" to start training.

<p align="center">
  <img src="https://github.com/anoushkajain/UnitRefine/blob/main/src/unitrefine/resources/model_training_example.jpg" width="500"/>
</p>

In the terminal, you will see logs from model training, for example:
```bash
Running RandomForestClassifier with imputation knn and scaling StandardScaler()
    Balanced Accuray: 
    Precision: 
    Recall: 
```
If your balanced accuracy is above 75%, your labeling is generally consistent and reliable. You can inspect the model output with the "Inspect" button.
If balanced accuracy is lower, we recommend relabeling the cluster and retraining the model to improve results. You can simply use the "Curate" button again to label additional clusters and train a new model to check if this improves the performance.

Alternatively, you can use retraining with active learning to automatically identify clusters with low model prediction confidence. This enhances the re-training efficency by selectively providing training data for clusters that the model can not predict already. You can set the percentage of clusters that should be staged for labeling (default is 20%) and press the "Relabel" button to curate low-confidence clusters.

<p align="center">
  <img src="https://github.com/anoushkajain/UnitRefine/blob/main/src/unitrefine/resources/re-label.JPG" width="500"/>
</p>

After curation, press the "Retrain model" button to train a new model that includes the additional low-confidence labels. This model will be saved under a new name, e.g. model_1_retrained_01. You can repeat relabeling and retraining as often as you want. Each step will increase the number of available labels and create a new model version (labeld model_1_retrained_02, etc) until a strong model has been found. In our experience model the performance should clearly improve when labeling about 10% of the curated clusters and the total number of curated clusters should be at least 50 or more. Once you have a trained model with good balanced accuracy you can use it create cluster labels for any recording.

## Loading Your Model

To make cluster predictions on any new or already loaded recording, you can use your trained models, load any existing model from your local drive or use a pretrained online model. All models that you trained in the current project are already available in the drop-down menu. 

<p align="center">
  <img src="https://github.com/anoushkajain/UnitRefine/blob/main/src/unitrefine/resources/predict-gui.JPG" width="500"/>
</p>

Use the '+ Load' button to additional models by selecting their respective folders, e.g. from a different project.  

Use the '+ Load from HFH' button to load a pretrained model from the Hugging Face Hub (HFH).
For example use the path below for a model that was trained on mouse Neuropixels recordings in diverse brain regions:

```bash
AnoushkaJain3/UnitRefine-mice-sua-classifier
```
You can find a collection of the existing pre-trained models [here](https://huggingface.co/AnoushkaJain3).
These contain all models from different species and recording settings that were used in the [UnitRefine paper](https://www.biorxiv.org/content/10.1101/2025.03.30.645770v2.full). When using a pre-trained binary classifier that predicts clusters as "good" versus "not-sua" (such as the UnitRefine-mice-sua-classifier), both "MUA" and "Noise" labels from manual curation will be automatically combined into "not-sua".

After selecting a model, use the "Inspect" button to check the predicted labels for each loaded recording. 
The model predictions are saved in the project folder under: `analyzer_folder/labels_from_UnitRefine-mice-sua-classifier.csv`

You should see a good relation between the human-inferred cluster label and the model prediction in the column 'model'.
You can also sort the list by model confidence to see high confidence clusters at the top. If high confidence clusters are incorrect or if you think the model needs overall improvement, you can relabel the units and retrain the model until the visual inspections confirms reliable prediction performance.

Lastly, use the 'Generate code' button at the bottom of the GUI to generate a short code segment in the console that you can use to automatically apply the model to a given recording as part of your analysis pipeline.

---

That is it! Now, you can know how to add sorting analyzers, curate the data, train a model, and validate its performance.  
Keep an eye on the feedback printed in the terminal—it provides helpful guidance throughout the process.

You can also generate Python code from the GUI and reuse it later in your own scripts.  

The next time you run **UnitRefine**, simply point to the same project folder and everything will be loaded automatically.

```bash
uv run unitrefine --project_folder my_existing_project
```

---
## FAQs

**1. Issues with installing UV**

For windows users trying to install uv, try doing 
```bash
pip install uv
```
if this doesn't work then type the following on your cmd. 
```bash
powershell -ExecutionPolicy ByPass -c "irm https://astral.sh/uv/install.ps1 | iex"
```

**2. Number of labels**

You need to provide at least 6 labels for each used class (SUA, MUA, Noise) to prevent errors when training a new model. You can also use only SUA and Noise labels to create a binary instead of a 3-class classifier. As a starting model with decent performance you should label at least 10% of the data (should be more than 50 clusters in total).

----
If you identify other problems, leave an issue in the repository or reach out to us via email.
