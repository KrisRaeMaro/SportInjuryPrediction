# SportInjuryPrediction

# README

## Overview
Scripts and notebooks for exploring, preprocessing, and modeling injury prediction in athletes using machine learning pipelines. The goal is to analyze the data, preprocess it for machine learning, and build predictive models using XGBoost and CNNs. Below is an explanation of the files included in this project.

## File Descriptions

### Notebooks

1. **DataExploration.ipynb**
   - Purpose: Perform exploratory data analysis (EDA) to understand the dataset's structure, distributions, and temporal trends.
   - Key Tasks:
     - Visualizations of key features.
     - Analysis of injury occurrences over time.
     - Identification of potential predictors of injury risk.

2. **InjuryModels.ipynb**
   - Purpose: Train and evaluate machine learning models for injury prediction.
   - Key Tasks:
     - Comparison of models (e.g., XGBoost, CNN).
     - Evaluation metrics (e.g., precision, recall, F1-score, AUC).
     - SHAP analysis for feature importance.

### Python Scripts

1. **preprocess.py**
   - Purpose: Handle data preprocessing tasks such as cleaning, scaling, and feature engineering.
   - Key Functions:
     - Scaling features.
     - Merging files
     - Generating new features

2. **utils.py**
   - Purpose: Provide utility functions for common tasks.
   - Key Functions:
     - Plotting helper functions (e.g., confusion matrix, ROC curves).
     - Data transformation utilities.

3. **mts_cnn_pipeline.py**
   - Purpose: Implement a multivariate time-series CNN pipeline for athlete-specific injury prediction.
   - Key Features:
     - CNN architecture with convolutional, pooling, and dense layers.
     - Batch normalization and dropout for regularization.
     - Evaluation using metrics like precision, recall, and AUC.

4. **xgboost_pipeline.py**
   - Purpose: Train and evaluate an XGBoost-based injury prediction pipeline.
   - Key Features:
     - Cross-validation and testing.
     - Handling class imbalance with SMOTE.
     - SHAP analysis for feature importance.


### Prerequisites
- Python 3.8 or higher.
- Required libraries:
  - `pandas`
  - `numpy`
  - `matplotlib`
  - `seaborn`
  - `xgboost`
  - `scikit-learn`
  - `shap`
  - `imblearn`



## Key Outputs
- Visualizations of injury trends and feature distributions.
- Trained models with evaluation metrics (e.g., AUC, precision, recall).
- Feature importance plots using SHAP.

## Future Work
- Improve handling of class imbalance.
- Experiment with additional machine learning models.
- Incorporate new features (e.g., heart rate or external workload).

## Contact
For any questions, please contact:
- **Name:** Kris Marosi
- **Email:** Krismarosi@gmail.com
