# Model Training Tutorial

This tutorial demonstrates how to train custom machine learning models for automated spike sorting curation using UnitRefine. The training process helps create classifiers that can automatically distinguish between high-quality neural units and noise/artifacts.


### Choose Your Training Method

**Method 1: CSV Files** (`train_new_model_using_csv.ipynb`)
- ✅ Use if you have pre-computed quality metrics as CSV files
- ✅ Ideal for combining data from multiple sessions
- ✅ Works with any spike sorting software output

**Method 2: SortingAnalyzer** (`train_model_using_sorting_analyzer.ipynb`)  
- ✅ Use if working directly with SpikeInterface
- ✅ Automatically computes quality and template metrics
- ✅ More integrated workflow with SpikeInterface ecosystem

