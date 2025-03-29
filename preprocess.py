import os
import glob
import pandas as pd
from sklearn.preprocessing import MinMaxScaler, StandardScaler, RobustScaler

class PreprocessingPipeline:
    """
    Preprocessing pipeline for injury, game_workload, and metrics datasets.
    Validates datasets, merges them, scales features per athlete, and calculates ACWR per athlete.
    """
    def __init__(self, directory, scaler='MinMaxScaler', acute_span=7, chronic_span=28, workload_column='game_workload'):
        self.directory = directory
        self.scaler = scaler
        self.acute_span = acute_span
        self.chronic_span = chronic_span
        self.workload_column = workload_column
        self.dfs = {}
        self.final_df = None

    # ------------------------------
    #  Data Loading and Parsing
    # ------------------------------
    def load_csv_files(self):
        """Load all CSV files from the given directory and convert date columns."""
        if not os.path.exists(self.directory):
            raise FileNotFoundError(f"The specified directory does not exist: {self.directory}")
        
        for file in glob.glob(os.path.join(self.directory, '*.csv')):
            file_name = os.path.splitext(os.path.basename(file))[0]
            df = pd.read_csv(file, low_memory=False)
            
            # Date Conversion
            date_columns = [col for col in df.columns if 'date' in col.lower()]
            for col in date_columns:
                df[col] = pd.to_datetime(df[col], errors='coerce')
                if df[col].isna().sum() > 0:
                    print(f"{file_name}: {df[col].isna().sum()} invalid date entries in '{col}' were set to NaT.")
            
            self.dfs[file_name] = df
            print(f"{file_name}: {df.shape[0]} rows, {df.shape[1]} columns, missing entries: {df.isnull().sum().sum()}")
            print(f"{file_name}: Data on {df['athlete_id'].nunique()} athletes")

    # ------------------------------
    # Data Validation - Check Var Overlap
    # ------------------------------
    def validate_data(self):
        """Check consistency between datasets."""
        d_i = self.dfs.get('injuries', pd.DataFrame()).copy()
        d_w = self.dfs.get('game_workload', pd.DataFrame()).copy()
        d_m = self.dfs.get('metrics', pd.DataFrame()).copy()
        
        injury_pairs = set(d_i[['date', 'athlete_id']].apply(tuple, axis=1))
        workload_pairs = set(d_w[['date', 'athlete_id']].apply(tuple, axis=1))
        
        if not injury_pairs.issubset(workload_pairs):
            missing_injury_pairs = injury_pairs - workload_pairs
            print(f"Missing injury info in workload dataset: {sorted(missing_injury_pairs)}")
        
        hip_metrics = d_m[d_m['metric'] == 'hip_mobility']
        hip_metric_pairs = set(hip_metrics[['date', 'athlete_id']].apply(tuple, axis=1))
        
        if not workload_pairs.issubset(hip_metric_pairs):
            missing_workload_pairs = workload_pairs - hip_metric_pairs
            print(f"Missing workload info in hip_mobility metrics dataset: {sorted(missing_workload_pairs)}")

    # ------------------------------
    #  Data Preparation
    # ------------------------------
    def prepare_data(self):
        """Merge datasets into a single table and add session numbers and game day indicator"""
        d_i = self.dfs.get('injuries', pd.DataFrame()).copy()
        d_m = self.dfs.get('metrics', pd.DataFrame()).copy()
        d_w = self.dfs.get('game_workload', pd.DataFrame()).copy()
 
        d_i['injured'] = 1
        d_w['game_day'] = 1
        
        # Merge workload + injury datasets
        dt = pd.merge(d_w, d_i, on=['athlete_id', 'date'], how='left')
        dt['injured'] = dt['injured'].fillna(0).astype(int)
        dt['game_day'] = dt['game_day'].fillna(0).astype(int)
        
        # Pivot metrics data
        metrics_pivot = d_m.pivot(index=['athlete_id', 'date'], columns='metric', values='value').reset_index()
        m = pd.merge(metrics_pivot, dt[['athlete_id', 'date', 'game_workload', 'game_day', 'injured']], on=['athlete_id', 'date'], how='left')

        # Add session numbers
        # m['session_number'] = m.groupby('athlete_id').cumcount() + 1
        self.final_df = m

    def add_previous_injury(self):
        """Add 'previous_injury' column based on the first injury date."""
        self.final_df['injured'] = self.final_df['injured'].fillna(0)
        self.final_df['previous_injury'] = 0
        
        for athlete_id, group in self.final_df.groupby('athlete_id'):
            first_injury_date = group.loc[group['injured'] == 1, 'date'].min()
            if pd.notnull(first_injury_date):
                self.final_df.loc[
                    (self.final_df['athlete_id'] == athlete_id) & (self.final_df['date'] >= first_injury_date),
                    'previous_injury'
                ] = 1

    # ------------------------------
    # Feature Scaling (Per Athlete)
    # ------------------------------
    def scale_features(self, features_to_scale):
        """Scale specified features per athlete."""
        scaler_map = {
            'MinMaxScaler': MinMaxScaler(),
            'StandardScaler': StandardScaler(),
            'RobustScaler': RobustScaler()
        }
        scaler = scaler_map.get(self.scaler, MinMaxScaler())
        print(f"# of missing data points: {self.final_df[features_to_scale].isnull().sum()}, Imputing missing ")
        
        #impute missing
        self.final_df[features_to_scale] = self.final_df[features_to_scale].bfill()
        self.final_df[features_to_scale] = self.final_df[features_to_scale].fillna(self.final_df[features_to_scale].mean())

        scaled_data = []
        
        for athlete_id, group in self.final_df.groupby('athlete_id'):
            group = group.copy()
            scaled_features = pd.DataFrame(
                scaler.fit_transform(group[features_to_scale]),
                columns=[f"{col}_scaled" for col in features_to_scale]
            )
            group.reset_index(drop=True, inplace=True)
            scaled_features.reset_index(drop=True, inplace=True)
            group = pd.concat([group, scaled_features], axis=1)
            scaled_data.append(group)

        self.final_df = pd.concat(scaled_data, ignore_index=True)

    # ------------------------------
    # ACWR Calculation (Per Athlete)
    # ------------------------------
    def calculate_acwr(self):
        """Calculate Acute and Chronic Workloads and ACWR per athlete."""
        acwr_data = []
        
        for athlete_id, group in self.final_df.groupby('athlete_id'):
            group = group.sort_values(by='date').reset_index(drop=True)
            
            # Acute and Chronic Workloads
            group['acute_workload'] = group[self.workload_column].ewm(span=self.acute_span, adjust=False).mean()
            group['chronic_workload'] = group[self.workload_column].ewm(span=self.chronic_span, adjust=False).mean()
            
            # Acute:Chronic Workload Ratio (ACWR)
            group['acute_chronic_ratio'] = group['acute_workload'] / (group['chronic_workload'] + 1e-8)
            
            acwr_data.append(group)
        
        self.final_df = pd.concat(acwr_data, ignore_index=True)

    # ------------------------------
    #  Run Pipeline
    # ------------------------------
    def run(self):
        """Execute the full pipeline."""
        print("Loading datasets...")
        self.load_csv_files()
        
        print("Validating datasets...")
        self.validate_data()
        
        print("Preparing data...")
        self.prepare_data()
        
        print("Adding previous injury feature...")
        self.add_previous_injury()
        
        print("Scaling features per athlete...")
        self.scale_features(['game_workload', 'groin_squeeze', 'hip_mobility'])
        
        print("Calculating ACWR per athlete...")
        self.calculate_acwr()
        
        self.final_df.to_csv('data_acwr.csv', index=False)
        print("Data processing completed. Dataset saved as 'data_acwr.csv'.")
