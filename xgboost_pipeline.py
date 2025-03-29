import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import shap
import xgboost as xgb
from sklearn.model_selection import train_test_split
from sklearn.impute import SimpleImputer
from sklearn.metrics import (
    confusion_matrix, roc_auc_score, precision_score, recall_score, f1_score,
    ConfusionMatrixDisplay, roc_curve,precision_recall_curve
)
from imblearn.over_sampling import SMOTE
sns.set(style="whitegrid", rc={"axes.grid": False})

import warnings
warnings.filterwarnings('ignore')

# ------------------------------
# Sliding Window Data Prep
# ------------------------------

def prepare_sliding_window_data(df, features, history_window, future_window):
    """
    Create sliding windows from time-series data for modeling.
    """
    mts_data = []
    labels = []

    for athlete_id, group in df.groupby('athlete_id'):
        group = group.sort_values('date').reset_index(drop=True)

        for i in range(len(group) - history_window - future_window + 1):
            history_window_data = group[features].iloc[i:i+history_window].values
            future_window_data = group['injured'].iloc[i+history_window:i+history_window+future_window]

            label = int(future_window_data.sum() > 0)

            mts_data.append(history_window_data)
            labels.append(label)

    mts_data = np.array(mts_data)
    labels = np.array(labels)

    print(f"Sliding Window Data Shape: {mts_data.shape}")
    print(f"Labels Shape: {labels.shape}")

    return mts_data, labels


# ------------------------------
# Hold-Out Test Evaluation
# ------------------------------

def evaluate_on_test_set(model, X_test, y_test, threshold=0.3, feature_names=None):
    """
    Evaluate model on hold-out test set and generate metrics and plots.
    """
    print("Evaluating Model on Hold-Out Test Set...")
    y_pred_proba = model.predict_proba(X_test)[:, 1]
    y_pred = (y_pred_proba >= threshold).astype(int)

    # Calculate Metrics
    precision = precision_score(y_test, y_pred)
    recall = recall_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred)
    auc = roc_auc_score(y_test, y_pred_proba)
    print(f"Precision: {precision:.2f}, Recall: {recall:.2f}, F1: {f1:.2f}, AUC: {auc:.2f}")

    # Plot Confusion Matrix
    cm = confusion_matrix(y_test, y_pred)
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=['No Injury', 'Injury'])
    disp.plot(cmap='Blues')
    plt.title("Confusion Matrix")
    plt.show()

    # Plot ROC Curve
    fpr, tpr, _ = roc_curve(y_test, y_pred_proba)
    plt.figure(figsize=(4, 4))
    plt.plot(fpr, tpr, label='ROC curve (area = {:.2f})'.format(auc))
    plt.plot([0, 1], [0, 1], linestyle='--', color='gray', linewidth=1)
    plt.title('ROC Curve', fontsize=14, weight='bold')
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.grid(False)
    plt.legend()
    plt.show()

    #Plot Precision Recall
    precision_vals, recall_vals, _ = precision_recall_curve(y_test, y_pred_proba)
    plt.figure(figsize=(4, 4))
    plt.plot(recall_vals, precision_vals, marker='.')
    plt.title('Precision-Recall Curve')
    plt.xlabel('Recall')
    plt.ylabel('Precision')
    plt.grid(False)
    plt.xticks(fontsize=10)
    plt.yticks(fontsize=10)
    plt.show()

    # Feature Importance Plot
    if feature_names is not None and hasattr(model, 'feature_importances_'):
        importance_df = pd.DataFrame({
            'Feature': feature_names,
            'Importance': model.feature_importances_
        }).sort_values(by='Importance', ascending=False)
        plt.figure(figsize=(6, 6))
        sns.barplot(data=importance_df.head(20), x='Importance', y='Feature', palette='viridis')
        plt.title('Top 20 Feature Importance (XGBoost Native)')
        plt.gca().spines['top'].set_visible(False)
        plt.gca().spines['right'].set_visible(False)
        plt.gca().spines['left'].set_linewidth(1.2)
        plt.gca().spines['bottom'].set_linewidth(1.2)
        plt.xticks(fontsize=10)
        plt.yticks(fontsize=10)
        plt.show()
    else:
        print("Feature importances are not available or feature names are missing.")


# ------------------------------
# InjuryXGBoostModel Class
# ------------------------------


class InjuryXGBoostModel:
    def __init__(self, df: pd.DataFrame, config: dict):
        """
        Initialize the Model
        """
        print("Loading Model...")
        self.df = df
        self.config = config
        self.model = None
        print("Model initialized.\n")

    def process_data(self):
        """
        Prepare sliding window data and handle missing values.
        """
        print("Processing Data with Sliding Windows...")
        imputer = SimpleImputer(strategy='mean')
        self.df[self.config['FEATURES']] = imputer.fit_transform(self.df[self.config['FEATURES']])
        self.mts_data, self.labels = prepare_sliding_window_data(
            self.df, self.config['FEATURES'], self.config['HISTORY_WINDOW'], self.config['FUTURE_WINDOW']
        )
        self.mts_data_flat = self.mts_data.reshape(self.mts_data.shape[0], -1)


    def cross_validate(self, X_train, y_train):
        """
        Perform cross-validation using XGBoost's CV function.
        """
        print("Performing Cross-Validation...")
        dtrain = xgb.DMatrix(X_train, label=y_train)
        params = {
            'objective': 'binary:logistic',
            'eval_metric': 'auc',
            'random_state': self.config['RANDOM_STATE'],
            'scale_pos_weight': len(y_train) / sum(y_train)  # Adjust class weight
        }
        cv_results = xgb.cv(
            params,
            dtrain,
            num_boost_round=100,
            nfold=5,
            stratified=True,
            metrics='auc',
            early_stopping_rounds=10,
            seed=self.config['RANDOM_STATE'],
            as_pandas=True,
            verbose_eval=False
        )
        print("Cross-Validation Results:")
        print(cv_results)
        print(f"Mean AUC: {cv_results['test-auc-mean'].mean():.2f}")

        # Plot ROC Curve with Standard Deviation
        plt.figure(figsize=(6, 6))
        mean_auc = cv_results['test-auc-mean']
        std_auc = cv_results['test-auc-std']
        rounds = range(len(mean_auc))
        plt.plot(rounds, mean_auc, label='Mean AUC')
        plt.fill_between(rounds, mean_auc - std_auc, mean_auc + std_auc, color='b', alpha=0.2, label=f"mean AUC ± std :{cv_results['test-auc-mean'].mean():.2f}")
        plt.title('Train Set AUC', fontsize=14)
        plt.xlabel('Boosting Rounds')
        plt.ylabel('AUC')
        plt.legend()
        plt.grid(False)
        plt.show()

    def train_and_evaluate(self):
        """
        Train the model and evaluate it on a hold-out test set.
        """
        print("Splitting Data into Training and Test Sets...")
        X_train, X_test, y_train, y_test = train_test_split(
            self.mts_data_flat, self.labels, test_size=0.2, random_state=self.config['RANDOM_STATE'], stratify=self.labels
        )

        # Balance training data with SMOTE
        smote = SMOTE(random_state=self.config['RANDOM_STATE'])
        X_train_res, y_train_res = smote.fit_resample(X_train, y_train)

        # Perform Cross-Validation
        self.cross_validate(X_train_res, y_train_res)

        # Train the model
        print("Training the Model...")
        self.model = xgb.XGBClassifier(
            random_state=self.config['RANDOM_STATE'],
            eval_metric='logloss',
            scale_pos_weight=len(y_train_res) / sum(y_train_res)  # Adjust class weight
        )
        self.model.fit(X_train_res, y_train_res)
        
        # Evaluate the model on the hold-out test set
        evaluate_on_test_set(self.model, X_test, y_test, threshold=0.3,  feature_names=self.get_feature_names())


    def shap_analysis(self):
        """
        Perform SHAP analysis.
        """
        print("Performing SHAP Analysis...")
        explainer = shap.Explainer(self.model, self.mts_data_flat)
        shap_values = explainer(self.mts_data_flat)

        # SHAP Summary Plot
        shap.summary_plot(shap_values, features=self.mts_data_flat, feature_names=self.get_feature_names())

        # SHAP Waterfall Plot for a Single Instance
        shap.waterfall_plot(shap.Explanation(
            values=shap_values[0].values,
            base_values=shap_values[0].base_values,
            feature_names=self.get_feature_names(),
            data=self.mts_data_flat[0]
        ))

        # SHAP Decision Plot
        shap.decision_plot(
            base_value=shap_values[0].base_values,
            shap_values=shap_values[0].values,
            features=self.mts_data_flat[0],
            feature_names=self.get_feature_names()
        )

    def get_feature_names(self):
        return [f"Day_{i}_{feature}" for i in range(self.config['HISTORY_WINDOW']) for feature in self.config['FEATURES']]



# ------------------------------
# Main
# ------------------------------
if __name__ == '__main__':

    
    model = InjuryXGBoostModel(df, CONFIG)
    model.process_data()
    model.train_and_evaluate()
    model.shap_analysis()

