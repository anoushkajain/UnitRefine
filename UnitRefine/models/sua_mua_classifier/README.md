---
language:
- en
pipeline_tag: tabular-classification
tags:
- Computational Neuroscience 
license: mit
---

## Model description
This model is part of the `UnitRefine` project.
The model is trained on 11 mice in V1, SC, and ALM using Neuropixels on mice.
Each recording was labeled by at least two people and in different combinations.
The agreement amongst labelers is 80%. 

# Intended use
Used to identify SUA clusters automatically in SpikeInterface.

# How to Get Started with the Model
This can be used to automatically identify SUA units in spike-sorted outputs. If you have a sorting_analyzer, it can be used as follows:

``` python
    from spikeinterface.curation import auto_label_units
    labels = auto_label_units(
        sorting_analyzer = sorting_analyzer,
        repo_id = "SpikeInterface/UnitRefine_sua_mua_classifier",
        trusted = ['numpy.dtype']
    )
```

# Authors

Anoushka Jain and Chris Halcrow