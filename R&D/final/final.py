import pandas as pd
import numpy as np
from scipy.stats import pearsonr, spearmanr, kendalltau
from statsmodels.tsa.stattools import grangercausalitytests

# Load your data into a DataFrame
data = {
    'Date': ['29-12-2023', '28-12-2023', '27-12-2023', '26-12-2023', '22-12-2023',
             '21-12-2023', '20-12-2023', '19-12-2023', '18-12-2023', '15-12-2023',
             '14-12-2023', '13-12-2023', '12-12-2023', '11-12-2023', '08-12-2023',
             '07-12-2023', '06-12-2023', '05-12-2023', '04-12-2023', '01-12-2023'],
    'Open': [634.1, 639, 635, 643.9, 646, 615, 627.9, 623.95, 610, 600, 605, 616.55, 655, 664.9, 669.7, 728.85, 845, 860, 878.35, 875.85],
    'High': [643, 640.5, 641.95, 644.9, 650.6, 641.05, 635.95, 626, 624, 609.45, 616.4, 616.75, 658, 667, 679.85, 728.85, 847, 867, 881.15, 884.15],
    'Low': [631.35, 631.15, 630.15, 630, 635.1, 606.05, 610, 618.3, 607, 599.05, 603.2, 592.7, 614.8, 651.2, 637, 650.45, 810.05, 837, 853.15, 867.2],
    'SentimentScore': [0.099726686, 0.140969518, 0.109632387, 0.083223249, 0.100586355, 0.120252741, 0.040800903, 0.039489122, 0.031248197, -0.001318436,
                       0.089623864, -0.052592593, 0.078489583, 0.048585859, 0.048961327, 0.080987314, 0.154762098, 0.128178845, 0.143720256, 0.114615686]
}

df = pd.DataFrame(data)
df['Date'] = pd.to_datetime(df['Date'], format='%d-%m-%Y')

# Pearson Correlation
pearson_open, _ = pearsonr(df['SentimentScore'], df['Open'])
pearson_high, _ = pearsonr(df['SentimentScore'], df['High'])
pearson_low, _ = pearsonr(df['SentimentScore'], df['Low'])

# Spearman Correlation
spearman_open, _ = spearmanr(df['SentimentScore'], df['Open'])
spearman_high, _ = spearmanr(df['SentimentScore'], df['High'])
spearman_low, _ = spearmanr(df['SentimentScore'], df['Low'])

# Kendall Correlation
kendall_open, _ = kendalltau(df['SentimentScore'], df['Open'])
kendall_high, _ = kendalltau(df['SentimentScore'], df['High'])
kendall_low, _ = kendalltau(df['SentimentScore'], df['Low'])

# Granger Causality
# Prepare data for Granger causality test
df_granger = df[['SentimentScore', 'Open']].dropna()
max_lag = 5  # maximum number of lags to consider
granger_test = grangercausalitytests(df_granger, max_lag, verbose=True)

# Results
print("Pearson Correlation (SentimentScore vs Open):", pearson_open)
print("Pearson Correlation (SentimentScore vs High):", pearson_high)
print("Pearson Correlation (SentimentScore vs Low):", pearson_low)

print("Spearman Correlation (SentimentScore vs Open):", spearman_open)
print("Spearman Correlation (SentimentScore vs High):", spearman_high)
print("Spearman Correlation (SentimentScore vs Low):", spearman_low)

print("Kendall Correlation (SentimentScore vs Open):", kendall_open)
print("Kendall Correlation (SentimentScore vs High):", kendall_high)
print("Kendall Correlation (SentimentScore vs Low):", kendall_low)

print("Granger Causality Test Results:", granger_test)
