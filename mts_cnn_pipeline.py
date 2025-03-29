import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import shap
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import (
    Conv1D, MaxPooling1D, GlobalAveragePooling1D, Dense, Dropout, BatchNormalization
)
from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    confusion_matrix, roc_auc_score, precision_score, recall_score, f1_score,
    roc_curve, precision_recall_curve, ConfusionMatrixDisplay
)
from imblearn.over_sampling import ADASYN
import warnings
from shap import Explanation, KernelExplainer
warnings.filterwarnings('ignore')
sns.set(style="whitegrid", rc={"axes.grid": False})

# ------------------------------
# Data Prep
# ------------------------------

def prepare_sliding_window_data(df, features, history_window, future_window, step):
    """
    Create sliding windows for time-series modeling.
    """
    mts_data = []
    labels = []

    for athlete_id, group in df.groupby('athlete_id'):
        group = group.sort_values('date').reset_index(drop=True)

        if group['injured'].sum() == 0:
            print(f"Skipping Athlete {athlete_id}: No injury records found.")
            continue

        for i in range(0, len(group) - history_window - future_window + 1, step):
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
# CNN Model
# ------------------------------

def create_cnn_model(input_shape):
    """
    Build a CNN model for time-series injury prediction.
    """
    model = Sequential([
        Conv1D(filters=32, kernel_size=3, activation='relu', input_shape=input_shape),
        BatchNormalization(),
        MaxPooling1D(pool_size=2),
        Conv1D(filters=64, kernel_size=3, activation='relu'),
        BatchNormalization(),
        MaxPooling1D(pool_size=2),
        GlobalAveragePooling1D(),
        Dropout(0.5),
        Dense(64, activation='relu'),
        Dense(1, activation='sigmoid')
    ])
    model.compile(
        optimizer='adam',
        loss='binary_crossentropy',
        metrics=['accuracy', 'AUC']
    )
    return model

# ------------------------------
# 4. Evaluation
# ------------------------------

def evaluate_model(model, X_test, y_test):
    """
    Evaluate the CNN model and generate visualizations.
    """
    print("Evaluating Model...")
    if len(np.unique(y_test)) < 2:
        print("Skipping Evaluation: Not enough class diversity in test data.")
        return

    y_pred = (model.predict(X_test) > 0.5).astype(int).ravel()
    y_pred_proba = model.predict(X_test).ravel()

    # Confusion Matrix
    # cm = confusion_matrix(y_test, y_pred)
    # disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=['No Injury', 'Injury'])
    # disp.plot(cmap='Blues')
    # plt.title("Confusion Matrix")
    # plt.show()

    precision = precision_score(y_test, y_pred)
    recall = recall_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred)
    auc = roc_auc_score(y_test, y_pred_proba)
    print(f"Precision: {precision:.2f}, Recall: {recall:.2f}, F1: {f1:.2f}, AUC: {auc:.2f}")
# ------------------------------
# 5. SHAP Analysis with KernelExplainer
# ------------------------------

def shap_analysis(model, X_test, feature_names, history_window):
    """
    Perform SHAP analysis for CNN model using KernelExplainer and save the most important feature.
    """
    print("Performing SHAP Analysis...")

    # Flatten time-series data for SHAP
    X_test_flat = X_test.reshape(X_test.shape[0], -1)

    # Flatten feature names
    flat_feature_names = [f"Day_{i}_{f}" for i in range(history_window) for f in feature_names]

    # Prediction Wrapper
    def model_predict(data):
        data = data.reshape((-1, history_window, len(feature_names)))
        return model.predict(data).flatten()

    # KernelExplainer
    explainer = KernelExplainer(model_predict, X_test_flat[:10])
    shap_values = explainer.shap_values(X_test_flat[:10])

    # Summary Plot
    shap.summary_plot(shap_values, features=X_test_flat[:10], feature_names=flat_feature_names)

    # Save Most Important Feature
    mean_abs_shap = np.abs(shap_values).mean(axis=0)
    top_feature_index = np.argmax(mean_abs_shap)
    top_feature_name = flat_feature_names[top_feature_index]
    top_feature_importance = mean_abs_shap[top_feature_index]

    print(f"Most Important Feature: {top_feature_name} (Importance: {top_feature_importance:.4f})")
    return top_feature_name, top_feature_importance



# ------------------------------
# Athlete-Specific CNN Class
# ------------------------------

class AthleteCNNModel:
    def __init__(self, df, config):
        self.df = df
        self.config = config
        self.models = {}
        self.top_features = {}
    
    def process_data(self):
        self.data = {}
        for athlete_id, group in self.df.groupby('athlete_id'):
            if group['injured'].sum() == 0:
                print(f"Skipping Athlete {athlete_id}: No injury records found.")
                continue

            self.data[athlete_id] = prepare_sliding_window_data(
                group,
                self.config['FEATURES'],
                self.config['HISTORY_WINDOW'],
                self.config['FUTURE_WINDOW'],
                self.config['WINDOW_STEP']
            )
    
    def train(self):
      for athlete_id, (X, y) in self.data.items():
        print(f"🚀 Training model for Athlete {athlete_id}...")

        # Ensure sufficient samples in each class
        class_counts = np.bincount(y)
        if len(class_counts) < 2 or min(class_counts) < 2:
            print(f"Skipping Athlete {athlete_id}: Not enough samples for stratified splitting.")
            continue

        # Oversample minority class with ADASYN and fallback to RandomOverSampler
        try:
            oversampler = ADASYN(random_state=self.config['RANDOM_STATE'])
            X, y = oversampler.fit_resample(X.reshape(X.shape[0], -1), y)
        except ValueError as e:
            print(f"ADASYN failed for Athlete {athlete_id}, falling back to RandomOverSampler: {e}")
            from imblearn.over_sampling import RandomOverSampler
            oversampler = RandomOverSampler(random_state=self.config['RANDOM_STATE'])
            X, y = oversampler.fit_resample(X.reshape(X.shape[0], -1), y)

        X = X.reshape(-1, self.config['HISTORY_WINDOW'], len(self.config['FEATURES']))

        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=self.config['RANDOM_STATE'], stratify=y
        )

        model = create_cnn_model((self.config['HISTORY_WINDOW'], len(self.config['FEATURES'])))
        model.fit(X_train, y_train, epochs=self.config['EPOCHS'], batch_size=self.config['BATCH_SIZE'], verbose=1)
        self.models[athlete_id] = model

        evaluate_model(model, X_test, y_test)
        top_feature, importance = shap_analysis(
                model, X_test, self.config['FEATURES'], self.config['HISTORY_WINDOW'])

        self.top_features[athlete_id] = {'feature': top_feature, 'importance': importance}

    
    def save_top_features(self, output_path='top_features_per_athlete.csv'):
        """
        Save the most important feature for each athlete to a CSV file.
        """
        print("Saving Top Features for Each Athlete...")
        top_features_df = pd.DataFrame.from_dict(self.top_features, orient='index')
        top_features_df.reset_index(inplace=True)
        top_features_df.rename(columns={'index': 'athlete_id'}, inplace=True)
        top_features_df.to_csv(output_path, index=False)


# ------------------------------
# Main 
# ------------------------------

if __name__ == '__main__':
    model = AthleteCNNModel(df, CONFIG)
    model.process_data()
    model.train()
    model.save_top_features()
