# Model Training Tutorial

This tutorial demonstrates how to train custom machine learning models for automated spike sorting curation using UnitRefine. The training process helps create classifiers that can automatically distinguish between high-quality neural units and noise/artifacts.

## 📁 Tutorial Structure

```
train_model/
├── data/                                    # All data files consolidated here
│   ├── quality_metrics.csv                # Example quality metrics for training
│   ├── cluster_group.tsv                  # Manual curation labels (good/mua/noise)
│   ├── new_data.csv                       # Example dataset for model application  
│   ├── best_model.skops                   # Trained model (generated after training)
│   ├── model_accuracies.csv               # Performance metrics (generated)
│   ├── model_info.json                    # Model metadata (generated)
│   ├── training_data.csv                  # Training features (generated)
│   └── model_predictions.csv              # Predictions on new data (generated)
├── train_new_model_using_csv.ipynb        # Training from CSV files
├── train_model_using_sorting_analyzer.ipynb # Training from SpikeInterface analyzers
└── README.md                              # This file
```

## 🚀 Getting Started

### Prerequisites

- **UnitRefine** installed and working
- **SpikeInterface** (latest version recommended)
- **Python packages**: pandas, numpy, matplotlib, scikit-learn
- **Manual curation data** or quality metrics from your spike sorting

### Choose Your Training Method

**Method 1: CSV Files** (`train_new_model_using_csv.ipynb`)
- ✅ Use if you have pre-computed quality metrics as CSV files
- ✅ Ideal for combining data from multiple sessions
- ✅ Works with any spike sorting software output

**Method 2: SortingAnalyzer** (`train_model_using_sorting_analyzer.ipynb`)  
- ✅ Use if working directly with SpikeInterface
- ✅ Automatically computes quality and template metrics
- ✅ More integrated workflow with SpikeInterface ecosystem

## 📊 Tutorial 1: Training from CSV Files

### Overview
Learn to train models using pre-computed quality metrics and manual curation labels stored in CSV/TSV files.

### Key Features
- **Multi-dataset training**: Combine data from multiple recording sessions
- **Flexible input**: Works with any quality metrics format
- **Comprehensive evaluation**: Detailed performance analysis and feature importance

### Required Input Files
- `quality_metrics.csv`: Computed quality metrics for each unit
- `cluster_group.tsv`: Manual curation labels in Phy format
- `new_data.csv`: New dataset for testing the trained model

### Workflow
1. **Load Data**: Import quality metrics and curation labels
2. **Data Processing**: Clean and prepare data for training  
3. **Model Training**: Test multiple ML algorithms and configurations
4. **Performance Analysis**: Evaluate accuracy, precision, recall
5. **Feature Importance**: Understand which metrics matter most
6. **Apply Model**: Use trained model on new, unseen data

## 🔬 Tutorial 2: Training from SortingAnalyzer

### Overview  
Train models directly from SpikeInterface SortingAnalyzer objects with automated metric computation.

### Key Features
- **Automated metrics**: Computes quality and template metrics automatically
- **Simulated data**: Uses realistic synthetic data for demonstration
- **Full pipeline**: From raw data to trained model

### Workflow
1. **Generate Data**: Create realistic synthetic spike sorting data
2. **Compute Metrics**: Calculate comprehensive quality metrics
3. **Manual Labels**: Create example curation decisions
4. **Train Model**: Test multiple classifier configurations  
5. **Analyze Results**: Examine performance and feature importance

## 🎯 Model Training Process

### Algorithms Tested
- **Random Forest**: Robust, interpretable, handles missing values
- **Gradient Boosting**: High accuracy, good for complex patterns
- **Logistic Regression**: Fast, interpretable, good baseline
- **Multi-layer Perceptron**: Neural network for non-linear patterns

### Data Preprocessing
- **Missing Value Imputation**: Median, mean, or mode strategies
- **Feature Scaling**: StandardScaler, RobustScaler, MinMaxScaler
- **Cross-validation**: Ensures robust performance estimates

### Quality Metrics Used
- **Signal Quality**: SNR, amplitude metrics, template properties
- **Contamination**: ISI violations, refractory period violations  
- **Stability**: Drift metrics, firing rate consistency
- **Isolation**: Cluster separation, silhouette scores

## 📈 Understanding Results

### Performance Metrics
- **Balanced Accuracy**: Accounts for class imbalance
- **Precision**: Fraction of predicted good units that are actually good
- **Recall**: Fraction of actual good units correctly identified
- **Feature Importance**: Which metrics contribute most to decisions

### Output Files
- `best_model.skops`: Trained model in secure skops format
- `model_accuracies.csv`: Performance comparison of all tested configurations
- `model_info.json`: Model metadata and training parameters
- `training_data.csv`: Feature matrix used for training
- `model_predictions.csv`: Predictions and confidence scores for new data

## 🔧 Customization

### Using Your Own Data

**For CSV Method:**
1. Replace `quality_metrics.csv` with your computed metrics
2. Replace `cluster_group.tsv` with your manual curation labels
3. Ensure column names match expected format
4. Update `metrics_cols` list if needed

**For SortingAnalyzer Method:**
1. Replace simulated data with your actual recordings
2. Load your sorting results instead of synthetic data
3. Provide your manual curation labels
4. Adjust quality metric computation parameters

### Advanced Configuration

```python
# Customize training parameters
trainer = train_model(
    mode="csv",  # or "analyzers"
    labels=your_labels,
    metrics_paths=your_csv_files,  # or analyzers=your_analyzers
    folder="my_model",
    
    # Algorithm selection
    classifiers=["RandomForestClassifier", "GradientBoostingClassifier"],
    
    # Data preprocessing  
    imputation_strategies=["median", "mean"],
    scaling_techniques=["standard_scaler", "robust_scaler"],
    
    # Feature selection
    metric_names=custom_metrics_list,  # or None for all metrics
    
    # Performance
    overwrite=True
)
```

## ✅ Best Practices

### Data Quality
- **Diverse Training Data**: Include multiple animals, brain regions, conditions
- **Balanced Labels**: Ensure good representation of both good and bad units  
- **Quality Control**: Verify manual curation consistency
- **Sufficient Sample Size**: At least 100-200 units per class recommended

### Model Selection
- **Cross-validation**: Always use proper train/validation/test splits
- **Feature Engineering**: Consider unit-specific vs. population metrics
- **Regularization**: Prevent overfitting with appropriate parameters
- **Interpretability**: Analyze feature importance for biological insights

### Deployment
- **Version Control**: Track model versions and training data
- **Documentation**: Record training parameters and performance
- **Validation**: Test on independent datasets before production use
- **Monitoring**: Track model performance over time

## 🚨 Troubleshooting

### Common Issues

**"File not found" errors:**
- Ensure all data files are in the `data/` folder
- Check file names and paths match exactly
- Verify files are not corrupted or empty

**Training failures:**
- Check for missing values in metrics or labels
- Ensure sufficient training data (>50 units minimum)
- Verify label format (binary: 0/1 or categorical: good/mua/noise)

**Poor performance:**
- Try different algorithms and preprocessing strategies
- Check data quality and manual curation consistency
- Consider feature selection or engineering
- Ensure training data represents target conditions

**Memory issues:**
- Reduce number of algorithms tested simultaneously
- Use fewer cross-validation folds
- Process datasets in smaller batches

## 📚 Additional Resources

- **UnitRefine Documentation**: Comprehensive usage guides
- **SpikeInterface Tutorials**: Learn about quality metric computation  
- **scikit-learn Documentation**: Understanding ML algorithms and preprocessing
- **Spike Sorting Best Practices**: Guidelines for manual curation

## 🤝 Contributing

Found issues or want to improve the tutorial? Please:
1. Report bugs via GitHub Issues
2. Suggest improvements or new features
3. Contribute example datasets
4. Share your trained models and results

---

**Need Help?** Check the troubleshooting section above or open an issue on GitHub.