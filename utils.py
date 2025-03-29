import matplotlib.pyplot as plt
from statsmodels.tsa.seasonal import STL
import numpy as np
import pandas as pd
import seaborn as sns
import warnings
from matplotlib.patches import FancyBboxPatch
sns.set(style="whitegrid", rc={"axes.grid": False}) 
warnings.filterwarnings('ignore')


def plot_injury_counts(data, date_column, injury_column, frequency='M', title='Injury Count Over Time'):
    """
    Create a bar plot of injury counts over a specified time frequency.
    """
    
    data = data.copy()
    data[date_column] = pd.to_datetime(data[date_column])
    injury_data = data[data[injury_column] == 1]

    # Resample by the specified frequency and count injuries
    injury_data['period'] = injury_data[date_column].dt.to_period(frequency)
    injury_count = injury_data.groupby('period').size().reset_index(name='injury_count')
    injury_count['period'] = injury_count['period'].dt.to_timestamp()

    total_injuries = injury_count['injury_count'].sum()
    print(f"N of total injuries: {total_injuries}")
    plt.figure(figsize=(9, 4))
    ax = sns.barplot(
        data=injury_count,
        x='period',
        y='injury_count',
        color='#5d6dff',
        edgecolor='black'
    )
    for bar in ax.patches:
        ax.annotate(
            f'{int(bar.get_height())}',
            (bar.get_x() + bar.get_width() / 2, bar.get_height()),
            ha='center',
            va='bottom',
            fontsize=10,
            color='black'
        )
    ax.set_xticks(range(len(injury_count['period'])))
    ax.set_xticklabels(
        injury_count['period'].dt.strftime('%Y-%m' if frequency == 'M' else '%Y-%W' if frequency == 'W' else '%Y-%d'),
        rotation=45,
        ha='right',
        fontsize=10
    )
    ax.grid(False)
    plt.title(title, fontsize=14, weight='bold')
    plt.xlabel('Time', fontsize=12)
    plt.ylabel('Number of Injuries', fontsize=12)
    plt.xticks(rotation=70, ha='center', fontsize=10)
    plt.tight_layout()
    plt.show()



def plot_kde(data, date_column, injury_column, frequency='W', title='Week'):
    """
    Create a KDE plot for injury frequency over time, aggregated by a specified frequency
    """
    data[date_column] = pd.to_datetime(data[date_column])
    injury_data = data[data[injury_column] == 1]
    injury_data['period'] = injury_data[date_column].dt.to_period(frequency)
    injury_counts = injury_data.groupby('period').size().reset_index(name='injury_count')

    injury_counts['period_numeric'] = injury_counts['period'].dt.to_timestamp().astype('int64') // 10**9
    injury_counts['period_label'] = injury_counts.index

    plt.figure(figsize=(5, 5))
    sns.kdeplot(
        data=injury_counts,
        x='period_label',
        y='injury_count',
        fill=True,
        thresh=0,
        levels=100,
        cmap='mako',
        clip=(
            (injury_counts['period_label'].min() - 2, injury_counts['period_label'].max() + 2),
            (0, injury_counts['injury_count'].max())
        )
    )
    plt.title(title, fontsize=14, weight='bold')
    plt.xlabel(f'{frequency.capitalize()}', fontsize=12)
    plt.ylabel('Injury Count', fontsize=12)

    xtick_labels = [
        f"{idx}"
        for idx, period in zip(injury_counts['period_label'], injury_counts['period'].astype(str))
    ]
    plt.xticks(
        ticks=injury_counts['period_label'][::4],  # Show every 4th label
        labels=xtick_labels[::4],  
        rotation=45,
        ha='right',
        fontsize=10
    )
    plt.grid(axis='y', linestyle='--', alpha=0.5)
    plt.tight_layout()
    plt.show()



def seasonality(data, date_column, target_column, frequency='W', seasonal_period=7, title="Injury Analysis"):
    """
    Perform STL decomposition on the target column aggregated by the given frequency.
    """

    data = data.sort_values(date_column).set_index(date_column)
    
    # Resample the data to the specified frequency
    aggregated = data[target_column].fillna(0).astype(int).resample(frequency).sum()
    
    #STL decomposition
    stl = STL(aggregated, seasonal=seasonal_period, robust=True)
    result = stl.fit()

    fig, axes = plt.subplots(2, 1, figsize=(10, 4), sharex=True)
    result.trend.plot(ax=axes[0], title='Trend', color='#5d6dff', linewidth=2)
    axes[0].grid(False)  
    result.seasonal.plot(ax=axes[1], title='Seasonality', color='#ff5d6d', linewidth=2)
    axes[1].grid(False)
    plt.suptitle(title, fontsize=16, weight='bold')
    plt.xlabel('Time', fontsize=12)
    plt.tight_layout()
    plt.show()

    # Variance Analysis
    seasonal_variance = np.var(result.seasonal)
    residual_variance = np.var(result.resid)
    variance_ratio = seasonal_variance / residual_variance

    print("Variance Analysis for Seasonality Significance")
    print(f"Seasonal Variance: {seasonal_variance:.4f}")
    print(f"Residual Variance: {residual_variance:.4f}")
    print(f"Variance Ratio (Seasonal/Residual): {variance_ratio:.2f}")

    if seasonal_variance > residual_variance:
        print("Seasonality is likely significant.")
    else:
        print("Residual variance dominates. Seasonality is weak or not significant.")


def plot_violin_by_injury(data, features, palette=None, figsize=(7,5)):
    """
    Create a violin plot to compare feature distributions by injury status.
    """
    if palette is None:
        palette = {'0.0': '#5d6dff', '1.0': '#ff5d6d'}

    long_df = pd.melt(
        data[features],
        id_vars=['date', 'injured'], 
        var_name='feature',        
        value_name='value'     
    )
    long_df['date'] = pd.to_datetime(long_df['date'])
    long_df['injured'] = long_df['injured'].astype(str)
    plt.figure(figsize=figsize)
    sns.violinplot(
        data=long_df,
        x='feature',
        y='value',
        hue='injured',
        split=True,
        scale='count',
        palette=palette
    )
    plt.title("Violin Plot of Feature Distributions by Injury Status", fontsize=14)
    plt.xlabel("Features", fontsize=12)
    plt.ylabel("Value", fontsize=12)
    plt.legend(title="Injury", loc="upper right", fontsize=10)
    plt.grid(axis='y', linestyle='--', alpha=0.5)
    plt.tight_layout()
    plt.xticks(rotation=45, fontsize=10)
    plt.show()


def plot_metrics(data, metric, outcome):
    """
    Plot a metric as a lineplot with injury markers overlaid in red.
    """
    data = data.sort_values('date').reset_index(drop=True)

    plt.figure(figsize=(14, 5))

    sns.lineplot(
        data=data,
        x='date',
        y=metric,
        label=f"{metric.capitalize()} Trend",
        color='#5d6dff',
        linewidth=2,
        marker='o',
    )
    injury_dates = data[data[outcome] == 1]['date']
    for injury_date in injury_dates:
        plt.axvline(x=injury_date, color='#ff5d6d', linestyle='--', alpha=0.1, label="Injury" if injury_date == injury_dates.iloc[0] else None)

    plt.title(f"{metric.replace('_', ' ').capitalize()} trend", fontsize=16)
    plt.xlabel("Date", fontsize=12)
    plt.ylabel(metric.replace('_', ' ').capitalize(), fontsize=12)
    plt.legend()
    plt.xticks(rotation=45, fontsize=10)
    plt.yticks(fontsize=10)
    plt.tight_layout()
    plt.show()

def plot_metrics_by_athlete(data, metric, outcome, athlete_id):
    """
    Plot a metric as a lineplot with injury markers overlaid in red / athlete.
    """
    data = data[data['athlete_id'] == athlete_id].sort_values('date').reset_index(drop=True)
    plt.figure(figsize=(14, 5))

    sns.lineplot(
        data=data,
        x='date',
        y=metric,
        label=f"{metric.capitalize()} Trend",
        color='#5d6dff',
        linewidth=2,
        marker='o',
    )
    injury_dates = data[data[outcome] == 1]['date']
    for injury_date in injury_dates:
        plt.axvline(x=injury_date, color='#ff5d6d', linestyle='--', alpha=0.8, label="Injury" if injury_date == injury_dates.iloc[0] else None)

    plt.title(f"{metric.replace('_', ' ').capitalize()} trend Athlete {athlete_id} ", fontsize=16)
    plt.xlabel("Date", fontsize=12)
    plt.ylabel(metric.replace('_', ' ').capitalize(), fontsize=12)
    plt.legend()
    plt.xticks(rotation=45, fontsize=10)
    plt.yticks(fontsize=10)
    plt.tight_layout()
    plt.show()


def aggregate_data(data, date_column, injury_column, features, period='W'):
    """
    Aggregate data by a unified period (weekly, monthly, etc.) across all athletes.
    """

    data[date_column] = pd.to_datetime(data[date_column])
    
    # Create a unified period column for consistent aggregation
    data['period'] = data[date_column].dt.to_period(period).dt.to_timestamp()
    
    aggregated_results = []
    
    # Group data by athlete
    for athlete_id, group in data.groupby('athlete_id'):
        group = group.sort_values(by=date_column).reset_index(drop=True)
        
        periods = group['period'].unique()
        
        print(f"Processing Athlete ID: {athlete_id}, Sequence Length: {len(periods)}")
        
        for i, current_period in enumerate(periods):
            period_data = group[group['period'] == current_period]
            
            # Assign injury status for the current period
            injured = 1 if (period_data[injury_column] == 1).any() else 0
            
            # Check next period's injury status
            if i + 1 < len(periods):
                next_period = periods[i + 1]
                next_period_data = group[group['period'] == next_period]
                injury_status_next_period = 1 if (next_period_data[injury_column] == 1).any() else 0
            else:
                injury_status_next_period = 0
            
            # Aggregate features
            aggregated_data = {
                'athlete_id': athlete_id,
                'period': current_period,
                'injured': injured,
                'injury_status_next_period': injury_status_next_period,
                'period_number': i + 1  
            }
            for feature in features:
                aggregated_data[f'{feature}_mean'] = period_data[feature].mean()
                aggregated_data[f'{feature}_std'] = period_data[feature].std()
                aggregated_data[f'{feature}_min'] = period_data[feature].min()
                aggregated_data[f'{feature}_max'] = period_data[feature].max()
                aggregated_data[f'{feature}_delta'] = period_data[feature].max() - period_data[feature].min()
            
            aggregated_results.append(aggregated_data)
    
    aggregated_df = pd.DataFrame(aggregated_results)
    return aggregated_df


def impute_missing(data, features):
    """
    Identify and impute missing values for each athlete, and print the percentage of missing data.
    """

    missing_info = {}
    for athlete_id, group in data.groupby('athlete_id'):
        # print(f"\nProcessing Athlete ID: {athlete_id}")

        missing_percentages = {}
        for feature in features:
            missing_count = group[feature].isna().sum()
            total_count = len(group)
            missing_percentage = (missing_count / total_count) * 100
            missing_percentages[feature] = missing_percentage
            # print(f"  {feature}: {missing_percentage:.2f}% missing")

        missing_info[athlete_id] = missing_percentages

        # impute
        for feature in features:
            if missing_percentages[feature] > 0:
                #TODO: forward-fill , backward-fill or other method
                group[feature] = group[feature].fillna(method='bfill').fillna(method='ffill')

        data.loc[group.index, features] = group[features]

    return data, missing_info


def plot_predictor_effects(results_df):
    """
    Plot the effects of predictors on injury risk with confidence intervals and significance highlighting.
    """
    # Sort predictors by coefficient estimates
    results_df = results_df.sort_values(by='coef', ascending=True).reset_index(drop=True)
    
    # Determine significance based on p-value
    results_df['color'] = results_df['pval'].apply(lambda p: '#5d6dff' if p < 0.05 else '#ebebeb')

    plt.figure(figsize=(8, 10))
    for i, row in results_df.iterrows():
        plt.errorbar(
            x=row['coef'], 
            y=row['predictor'], 
            xerr=[[row['coef'] - row['lower_ci']], [row['upper_ci'] - row['coef']]], 
            fmt='o', 
            color=row['color'], 
            markeredgecolor='black',
            capsize=5, 
            markersize=8, 
            label='_nolegend_'
        )
    
    plt.axvline(0, color='gray', linestyle='--', linewidth=1)
    plt.title("Weekly Metrics and Injury Risk", fontsize=14, weight='bold')
    plt.xlabel("Coefficient Estimate", fontsize=12)
    plt.ylabel("Predictor", fontsize=12)
    plt.grid(False)
    plt.yticks(fontsize=10)
    plt.tight_layout()
    plt.show()


def plot_sliding_windows(history_window=30, future_window=7, total_days=45, 
                         past_injury_days=None, future_injury_days=None, num_windows=5):
    """
    Plot sliding windows with history and prediction periods and injury markers.
    """
    if past_injury_days is None:
        past_injury_days = []
    if future_injury_days is None:
        future_injury_days = []

    fig, ax = plt.subplots(figsize=(12, 4))
    window_gap = 0.5  

    for i in range(num_windows): 
        y_position = num_windows - i * window_gap
        ax.broken_barh([(i, history_window)], (y_position, 0.4), facecolors='#5d6dff', 
                       edgecolor='black', label='Model Training Window' if i == 0 else "")
        ax.broken_barh([(i + history_window, future_window)], (y_position, 0.4), 
                       facecolors='#fff7aa', edgecolor='black', label='Prediction Window' if i == 0 else "")
        
        if i == 0: 
            for day in past_injury_days:
                if i <= day < i + history_window:
                    ax.plot(day, y_position + 0.2, 'wo', markersize=8, markeredgecolor='black', 
                            label='Past Injury Label' if day == past_injury_days[0] else "")

        if i == 2: 
            for day in future_injury_days:
                if i + history_window <= day < i + history_window + future_window:
                    ax.plot(day, y_position + 0.2, 'ro', markersize=8, 
                            label='Injury Label = 1' if day == future_injury_days[0] else "")
    ax.set_xlim(0, total_days)
    ax.set_ylim(2, num_windows + 1)  
    ax.set_xlabel('Days', fontsize=12)
    ax.set_ylabel('Sliding Windows', fontsize=12)
    ax.set_title('Sliding Window Framework', fontsize=14, weight='bold')
    ax.set_xticks(range(0, total_days + 1, 5))
    ax.set_yticks([])
    ax.legend(loc='upper right', fontsize=10)
    ax.grid(False)
    plt.tight_layout()
    plt.show()


def calculate_injury_statistics(df, athlete_column='athlete_id', date_column='date', injury_column='injured'):
    """
    Calculate detailed injury statistics including total injuries, time to first injury,
    average frequency of injuries, and injury frequency per athlete.
  
    """
    df[date_column] = pd.to_datetime(df[date_column])
    
    # Total injury count
    total_injuries = df[injury_column].sum()

    # Per-athlete start date and active days
    athlete_date_stats = (
        df.groupby(athlete_column)[date_column]
        .agg(start_date='min', end_date='max')
        .reset_index()
    )
    athlete_date_stats['days_active'] = (athlete_date_stats['end_date'] - athlete_date_stats['start_date']).dt.days

    # Number of injuries per athlete
    injury_counts = (
        df[df[injury_column] == 1]
        .groupby(athlete_column)
        .size()
        .reset_index(name='injury_count')
    )

    # Merge injury counts with athlete date stats
    athlete_stats = pd.merge(athlete_date_stats, injury_counts, on=athlete_column, how='left')
    athlete_stats['injury_count'] = athlete_stats['injury_count'].fillna(0)
    athlete_stats['injury_frequency_days'] = (
        athlete_stats['days_active'] / athlete_stats['injury_count'].replace(0, float('inf'))
    )

    # Time to first injury
    first_injury_dates = (
        df[df[injury_column] == 1]
        .groupby(athlete_column)[date_column]
        .min()
        .reset_index(name='first_injury_date')
    )
    athlete_stats = pd.merge(athlete_stats, first_injury_dates, on=athlete_column, how='left')
    athlete_stats['time_to_first_injury'] = (
        athlete_stats['first_injury_date'] - athlete_stats['start_date']
    ).dt.days.fillna(float('inf'))

    # Calculate global statistics
    avg_time_to_first_injury = athlete_stats['time_to_first_injury'].median()
    avg_injury_count_per_athlete = athlete_stats['injury_count'].median()
    mean_injury_frequency_days = athlete_stats.loc[
        athlete_stats['injury_count'] > 0, 'injury_frequency_days'
    ].mean()

    # Prepare results
    result = {
        'total_injuries': total_injuries,
        'avg_injury_count_per_athlete': avg_injury_count_per_athlete,
        'avg_time_to_first_injury_days': avg_time_to_first_injury,
        'mean_injury_frequency_days': mean_injury_frequency_days,
        'athlete_injury_statistics': athlete_stats,
    }
    
    return result


