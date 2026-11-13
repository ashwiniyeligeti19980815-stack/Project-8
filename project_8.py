
#Import Required Libraries

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
warnings.filterwarnings('ignore')

#Load the Dataset

try:
    # Attempting to load using the python engine and handling quoting anomalies
    df = pd.read_csv('C:\Users\ADMIN\OneDrive\Desktop\Project 8\all_upwork_jobs_2024-02-07-2024-03-24.csv')
except Exception as e:
    print(f"Standard read failed: {e}")
    # Fallback to a more permissive parsing configuration if still failing
    df = pd.read_csv('/content/all_upwork_jobs_2024-02-07-2024-03-24.csv', quoting=3, on_bad_lines='skip') # quoting=3 is csv.QUOTE_NONE

#Display the First 5 Rows

df.head()

##Check Dataset Information

df.info()

##Missing Values

df.isnull().sum()

#Rename the Columns Name

df = df.rename(columns={
    'are"': 'title',
    'https://www.upwork.com/jobs/Small-Steel-Business-Inventory-Management-System-Software_%7E01a8b4588424f63b85?source=rss': 'link',
    '2024-03-22 00:27:40+00:00': 'published_date',
    'False': 'is_hourly',
    'Unnamed: 4': 'min_budget',
    'Unnamed: 5': 'max_budget',
    '50.0': 'estimated_budget',
    'Australia': 'country'
})

df.head()

#Data Preparation

# 1. Convert published_date to datetime
df['published_date'] = pd.to_datetime(df['published_date'], errors='coerce')

# 2. Handle missing country values by filling with 'Unknown'
df['country'] = df['country'].fillna('Unknown')

# 3. Fill budget columns with 0 or NaN handling where appropriate
# We will keep them numeric and verify the results
df['min_budget'] = pd.to_numeric(df['min_budget'], errors='coerce')
df['max_budget'] = pd.to_numeric(df['max_budget'], errors='coerce')
df['estimated_budget'] = pd.to_numeric(df['estimated_budget'], errors='coerce')

# Display summary of cleaned dataset to verify changes
print(df.info())
display(df.head())

# Analyze the correlation between job title keywords and offered salaries

# 1. Define common keywords to search for in job titles
keywords_list = [
    'Developer', 'Designer', 'Writer', 'Manager', 'Marketing',
    'Video', 'Shopify', 'WordPress', 'Python', 'SEO',
    'Data', 'Social Media', 'Web', 'Mobile', 'Design'
]

# 2. Extract keyword matches and map to average/median estimated budget
keyword_salaries = []
for kw in keywords_list:
    # Filter jobs containing the keyword (case-insensitive) and extract estimated_budget
    subset = df[df['title'].str.contains(kw, case=False, na=False)]
    avg_budget = subset['estimated_budget'].mean()
    median_budget = subset['estimated_budget'].median()
    job_count = len(subset)
    if job_count > 0:
        keyword_salaries.append({
            'Keyword': kw,
            'Average Budget': avg_budget,
            'Median Budget': median_budget,
            'Job Count': job_count
        })

# 3. Create DataFrame and display the results
kw_df = pd.DataFrame(keyword_salaries).sort_values(by='Median Budget', ascending=False)
display(kw_df)

# 4. Plot the comparison of median budgets for the keywords
plt.figure(figsize=(12, 6))
sns.barplot(data=kw_df, x='Keyword', y='Median Budget', palette='viridis')
plt.title('Median Estimated Budget by Job Title Keyword')
plt.xlabel('Keyword')
plt.ylabel('Median Budget ($)')
plt.xticks(rotation=45)
plt.tight_layout()
plt.show()

#Task 2: Identify emerging job categories based on posting frequency

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# Ensure we have our clean dataset
# We will extract the week or day from 'published_date'
df['week'] = df['published_date'].dt.to_period('W').astype(str)

# Group by week and keyword to find trends
keywords_to_track = [
    'Developer', 'Designer', 'Writer', 'Manager', 'Marketing',
    'Video', 'Shopify', 'WordPress', 'Python', 'SEO',
    'Data', 'Social Media', 'Web', 'Mobile'
]

trend_data = []
for kw in keywords_to_track:
    subset = df[df['title'].str.contains(kw, case=False, na=False)]
    weekly_counts = subset.groupby('week').size().reset_index(name='count')
    weekly_counts['Keyword'] = kw
    trend_data.append(weekly_counts)

trends_df = pd.concat(trend_data, ignore_index=True)

# Pivot the data for easy viewing
trends_pivot = trends_df.pivot(index='week', columns='Keyword', values='count').fillna(0)

print("Weekly posting frequency by category:")
display(trends_pivot)

# Plot the trends over time for the top categories
plt.figure(figsize=(14, 7))
sns.lineplot(data=trends_df, x='week', y='count', hue='Keyword', marker='o')
plt.title('Job Posting Frequency Trends Over Time (Emerging Categories)')
plt.xlabel('Week')
plt.ylabel('Number of Postings')
plt.xticks(rotation=45)
plt.grid(True, linestyle='--', alpha=0.5)
plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
plt.tight_layout()
plt.show()

#Task 3: Predict high-demand job roles by analyzing job posting patterns over time

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

# Ensure our datetime field is active
df['date_only'] = df['published_date'].dt.date

# 1. Select the top high-demand roles from Task 2 (e.g., 'Web', 'Developer', 'Video')
target_keywords = ['Web', 'Developer', 'Video']

# Prepare historical daily frequency metrics
daily_trends = []
for kw in target_keywords:
    subset = df[df['title'].str.contains(kw, case=False, na=False)]
    daily_counts = subset.groupby('date_only').size().reset_index(name='postings_count')
    daily_counts['Keyword'] = kw
    daily_trends.append(daily_counts)

daily_df = pd.concat(daily_trends, ignore_index=True)
daily_df['date_only'] = pd.to_datetime(daily_df['date_only'])
daily_df = daily_df.sort_values(by=['Keyword', 'date_only']).reset_index(drop=True)

# 2. Build a Lag-based Feature set for time-series forecasting
# We will use the past 3 days of postings to predict tomorrow's posting count
forecast_data = []
for kw in target_keywords:
    kw_sub = daily_df[daily_df['Keyword'] == kw].copy()
    kw_sub['lag_1'] = kw_sub['postings_count'].shift(1)
    kw_sub['lag_2'] = kw_sub['postings_count'].shift(2)
    kw_sub['lag_3'] = kw_sub['postings_count'].shift(3)
    kw_sub['target'] = kw_sub['postings_count']
    # Drop incomplete rows
    kw_sub = kw_sub.dropna()
    forecast_data.append(kw_sub)

ml_df = pd.concat(forecast_data, ignore_index=True)

# Split into features (X) and target (y)
X = ml_df[['lag_1', 'lag_2', 'lag_3']]
y = ml_df['target']

# Split to Train/Test sets to evaluate metrics
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, shuffle=False)

# Train Random Forest Regressor
model = RandomForestRegressor(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

# Evaluate predictions
y_pred = model.predict(X_test)
mae = mean_absolute_error(y_test, y_pred)
mse = mean_squared_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)

print("### Forecasting Model Evaluation Metrics ###")
print(f"Mean Absolute Error (MAE): {mae:.2f} postings")
print(f"Mean Squared Error (MSE): {mse:.2f}")
print(f"R-squared Score (R2): {r2:.2f}")

# 3. Generate future forecast predictions and visualize
plt.figure(figsize=(14, 7))
colors = {'Web': 'blue', 'Developer': 'green', 'Video': 'orange'}

for kw in target_keywords:
    kw_data = ml_df[ml_df['Keyword'] == kw].copy()

    # Make prediction utilizing our trained model
    kw_data['predicted'] = model.predict(kw_data[['lag_1', 'lag_2', 'lag_3']])

    # Plot actual vs predicted
    plt.plot(kw_data['date_only'], kw_data['target'], label=f'{kw} (Actual)', color=colors[kw], alpha=0.6, marker='o')
    plt.plot(kw_data['date_only'], kw_data['predicted'], label=f'{kw} (Predicted Forecast)', color=colors[kw], linestyle='--', marker='x')

plt.title('Job Demand Forecasting: Actual vs Predicted Trends Over Time')
plt.xlabel('Date')
plt.ylabel('Daily Postings Count')
plt.grid(True, linestyle='--', alpha=0.5)
plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
plt.tight_layout()
plt.show()

# Task 4: Compare average hourly rates across different countries

# 1. Filter for hourly contracts
hourly_jobs = df[df['is_hourly'] == True].copy()

# Calculate an average hourly rate estimate per row from min and max budgets where available
hourly_jobs['hourly_rate_estimate'] = hourly_jobs[['min_budget', 'max_budget']].mean(axis=1)

# Fallback to estimated_budget if min and max are not populated
hourly_jobs['hourly_rate_estimate'] = hourly_jobs['hourly_rate_estimate'].fillna(hourly_jobs['estimated_budget'])

# Drop rows that still have no budget estimate
hourly_jobs = hourly_jobs.dropna(subset=['hourly_rate_estimate'])

# 2. Group by country and calculate the average and median hourly rates
country_rates = hourly_jobs.groupby('country').agg(
    avg_hourly_rate=('hourly_rate_estimate', 'mean'),
    median_hourly_rate=('hourly_rate_estimate', 'median'),
    job_count=('hourly_rate_estimate', 'count')
).reset_index()

# Filter for countries with a representative sample size (e.g., at least 5 postings)
country_rates_filtered = country_rates[country_rates['job_count'] >= 5].sort_values(by='avg_hourly_rate', ascending=False)

print("Average and Median Hourly Rates by Country (with >= 5 postings):")
display(country_rates_filtered)

# 3. Plot the top countries by average hourly rate
plt.figure(figsize=(14, 6))
sns.barplot(data=country_rates_filtered.head(15), x='country', y='avg_hourly_rate', palette='coolwarm')
plt.title('Top 15 Countries by Average Offered Hourly Rate (Minimum 5 Postings)')
plt.xlabel('Country')
plt.ylabel('Average Hourly Rate ($/hr)')
plt.xticks(rotation=45)
plt.grid(axis='y', linestyle='--', alpha=0.5)
plt.tight_layout()
plt.show()

"""#"""

# Task 6: Track changes in job market dynamics over months

# 1. Extract Year-Month for monthly dynamic tracking
df['month'] = df['published_date'].dt.to_period('M').astype(str)

# 2. Aggregate monthly metrics
monthly_dynamics = df.groupby('month').agg(
    total_postings=('title', 'count'),
    avg_estimated_budget=('estimated_budget', 'mean'),
    hourly_percentage=('is_hourly', lambda x: (x.sum() / len(x)) * 100)
).reset_index()

print("Monthly Job Market Dynamics:")
display(monthly_dynamics)

# 3. Build a Multi-Panel Dashboard to track the monthly shifts
fig, axes = plt.subplots(1, 3, figsize=(18, 5))

# Subplot 1: Total Job Postings Volume
sns.barplot(ax=axes[0], data=monthly_dynamics, x='month', y='total_postings', palette='Blues_d')
axes[0].set_title('Monthly Total Job Postings')
axes[0].set_xlabel('Month')
axes[0].set_ylabel('Number of Postings')
axes[0].grid(axis='y', linestyle='--', alpha=0.5)

# Subplot 2: Average Estimated Fixed-Price Budget
sns.lineplot(ax=axes[1], data=monthly_dynamics, x='month', y='avg_estimated_budget', marker='o', color='forestgreen', linewidth=2.5)
axes[1].set_title('Average Fixed-Price Budget Trend')
axes[1].set_xlabel('Month')
axes[1].set_ylabel('Average Budget ($)')
axes[1].grid(True, linestyle='--', alpha=0.5)

# Subplot 3: Hourly vs Fixed-Price Ratio (% Hourly Jobs)
sns.lineplot(ax=axes[2], data=monthly_dynamics, x='month', y='hourly_percentage', marker='s', color='crimson', linewidth=2.5)
axes[2].set_title('% of Jobs Offered as Hourly Contracts')
axes[2].set_xlabel('Month')
axes[2].set_ylabel('Hourly Contracts Proportion (%)')
axes[2].grid(True, linestyle='--', alpha=0.5)

plt.suptitle('Job Market Dynamics Dashboard (Monthly Tracking)', fontsize=16, weight='bold', y=1.05)
plt.tight_layout()
plt.show()

"""#"""

# Task 7: Investigate trends in the remote work landscape

# 1. Identify Remote Job Indicators within titles
remote_keywords = ['remote', 'wfh', 'work from home', 'anywhere', 'virtual', 'work online', 'home-based']

# Create boolean flag for matching remote indicators
df['is_explicitly_remote'] = df['title'].str.contains('|'.join(remote_keywords), case=False, na=False)

# 2. Extract metrics comparing remote vs. non-explicitly remote postings
remote_comparison = df.groupby('is_explicitly_remote').agg(
    total_jobs=('title', 'count'),
    hourly_percentage=('is_hourly', lambda x: (x.sum() / len(x)) * 100),
    avg_estimated_budget=('estimated_budget', 'mean'),
    median_estimated_budget=('estimated_budget', 'median')
).reset_index()

print("Comparison of Explicitly Remote vs. Unspecified/Traditional Jobs:")
display(remote_comparison)

# 3. Visualize the volumes and contract preferences side-by-side
fig, axes = plt.subplots(1, 2, figsize=(15, 6))

sns.barplot(ax=axes[0], data=remote_comparison, x='is_explicitly_remote', y='total_jobs', palette='Set2')
axes[0].set_title('Job Volume: Explicitly Remote vs Unspecified')
axes[0].set_xlabel('Explicitly Remote Label')
axes[0].set_ylabel('Number of Jobs')
axes[0].set_xticklabels(['Unspecified / Traditional', 'Explicitly Remote'])

sns.barplot(ax=axes[1], data=remote_comparison, x='is_explicitly_remote', y='avg_estimated_budget', palette='Set1')
axes[1].set_title('Average Fixed-Price Budgets')
axes[1].set_xlabel('Explicitly Remote Label')
axes[1].set_ylabel('Average Budget ($)')
axes[1].set_xticklabels(['Unspecified / Traditional', 'Explicitly Remote'])

plt.suptitle('Remote Work Landscape Analysis', fontsize=16, weight='bold')
plt.tight_layout()
plt.show()

# Task 8: Predict future job market trends (Daily-based forecasting to handle short dataset timeline)

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.ensemble import RandomForestRegressor

# 1. Aggregate job postings by day and keyword for forecasting
top_keywords = ['Web', 'Developer', 'Video', 'Designer']

df['date_only'] = df['published_date'].dt.date

daily_trends = []
for kw in top_keywords:
    subset = df[df['title'].str.contains(kw, case=False, na=False)].copy()
    daily_counts = subset.groupby('date_only').size().reset_index(name='postings')
    daily_counts['Keyword'] = kw
    daily_trends.append(daily_counts)

forecast_df = pd.concat(daily_trends, ignore_index=True)
forecast_df['date_only'] = pd.to_datetime(forecast_df['date_only'])
forecast_df = forecast_df.sort_values(by=['Keyword', 'date_only']).reset_index(drop=True)

# 2. Train a daily time-series forecasting model using lag variables
ml_days = []
for kw in top_keywords:
    kw_sub = forecast_df[forecast_df['Keyword'] == kw].copy()
    kw_sub['lag_1'] = kw_sub['postings'].shift(1)
    kw_sub['lag_2'] = kw_sub['postings'].shift(2)
    kw_sub['target'] = kw_sub['postings']
    ml_days.append(kw_sub)

ml_daily_df = pd.concat(ml_days, ignore_index=True).dropna()

# Train predictive models for each keyword separately
models = {}
for kw in top_keywords:
    kw_data = ml_daily_df[ml_daily_df['Keyword'] == kw]
    if len(kw_data) > 0:
        X_d = kw_data[['lag_1', 'lag_2']]
        y_d = kw_data['target']
        model_d = RandomForestRegressor(n_estimators=100, random_state=42)
        model_d.fit(X_d, y_d)
        models[kw] = model_d

# 3. Predict the next 5 days of demand under 3 scenarios
last_date = forecast_df['date_only'].max()
future_days = [last_date + pd.Timedelta(days=i) for i in range(1, 6)]
scenario_results = []

for kw in top_keywords:
    # Get last 2 known values to start recursive forecast
    kw_history = list(forecast_df[forecast_df['Keyword'] == kw]['postings'].tail(2).values)

    if len(kw_history) < 2 or kw not in models:
        continue

    last_2 = kw_history[-2]
    last_1 = kw_history[-1]

    predictions = []
    for _ in range(5):
        pred = models[kw].predict([[last_1, last_2]])[0]
        predictions.append(pred)
        last_2 = last_1
        last_1 = pred

    # Generate scenarios
    for i, day in enumerate(future_days):
        base_val = predictions[i]
        scenario_results.append({
            'date_only': day,
            'Keyword': kw,
            'Base Case': base_val,
            'Optimistic (+15%)': base_val * 1.15,
            'Pessimistic (-15%)': base_val * 0.85
        })

future_df = pd.DataFrame(scenario_results)
print("Predicted Future Daily Demand Scenarios (Next 5 Days):")
display(future_df.head(10))

# 4. Plot predictive scenario dashboard (Daily Timeline)
fig, axes = plt.subplots(2, 2, figsize=(16, 12))
axes = axes.flatten()

for idx, kw in enumerate(top_keywords):
    ax = axes[idx]
    # Historical daily data
    hist_data = forecast_df[forecast_df['Keyword'] == kw]
    ax.plot(hist_data['date_only'], hist_data['postings'], label='Historical Daily Actuals', color='black', marker='o', linewidth=2)

    # Future daily scenario projections
    if 'future_df' in globals() and not future_df.empty:
        fut_data = future_df[future_df['Keyword'] == kw]
        ax.plot(fut_data['date_only'], fut_data['Base Case'], label='Base Case Forecast', color='blue', linestyle='--', marker='x')
        ax.fill_between(fut_data['date_only'], fut_data['Pessimistic (-15%)'], fut_data['Optimistic (+15%)'],
                        color='blue', alpha=0.15, label='Volatility Range (Optimistic/Pessimistic)')

    ax.set_title(f'{kw} Daily Postings: Predictive Scenario Analysis')
    ax.set_xlabel('Timeline')
    ax.set_ylabel('Daily Job Openings')
    ax.grid(True, linestyle='--', alpha=0.5)
    ax.legend(loc='upper left')
    ax.tick_params(axis='x', rotation=30)

plt.suptitle('Job Market Trends Daily Forecast & Scenario Planning', fontsize=18, weight='bold', y=0.98)
plt.tight_layout()
plt.show()

