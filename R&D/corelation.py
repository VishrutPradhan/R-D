import pandas as pd

# Sample Data
data = {
    'Date': ['29-12-2023', '28-12-2023', '27-12-2023', '26-12-2023', '22-12-2023',
             '21-12-2023', '20-12-2023', '19-12-2023', '18-12-2023', '15-12-2023',
             '14-12-2023', '13-12-2023', '12-12-2023', '11-12-2023', '08-12-2023',
             '07-12-2023', '06-12-2023', '05-12-2023', '04-12-2023', '01-12-2023'],
    'Open': [634.1, 639, 635, 643.9, 646, 615, 627.9, 623.95, 610, 600, 605, 616.55, 655, 664.9, 669.7, 728.85, 845, 860, 878.35, 875.85],
    'High': [643, 640.5, 641.95, 644.9, 650.6, 641.05, 635.95, 626, 624, 609.45, 616.4, 616.75, 658, 667, 679.85, 728.85, 847, 867, 881.15, 884.15],
    'Low': [631.35, 631.15, 630.15, 630, 635.1, 606.05, 610, 618.3, 607, 599.05, 603.2, 592.7, 614.8, 651.2, 637, 650.45, 810.05, 837, 853.15, 867.2],
    'SentimentScore': [0.218584, 0.152988764, 0.120662791, 0.066329487, 0.272898824, 0.167446739, 0.093438776, 0.087944186, 0.180419883, 0.023959854, 0.25485, -0.12139, 0.052115, -0.15255, 0.266921348, 0.128692771, 0.168786207, 0.247234483, 0.264332584, 0.150920455]
}

# Create DataFrame
df = pd.DataFrame(data)

# Calculate Pearson correlation
pearson_corr_open = df['Open'].corr(df['SentimentScore'], method='pearson')
pearson_corr_high = df['High'].corr(df['SentimentScore'], method='pearson')
pearson_corr_low = df['Low'].corr(df['SentimentScore'], method='pearson')

# Calculate Spearman correlation
spearman_corr_open = df['Open'].corr(df['SentimentScore'], method='spearman')
spearman_corr_high = df['High'].corr(df['SentimentScore'], method='spearman')
spearman_corr_low = df['Low'].corr(df['SentimentScore'], method='spearman')

# Calculate Kendall correlation
kendall_corr_open = df['Open'].corr(df['SentimentScore'], method='kendall')
kendall_corr_high = df['High'].corr(df['SentimentScore'], method='kendall')
kendall_corr_low = df['Low'].corr(df['SentimentScore'], method='kendall')

print(f"Pearson correlation between Open and Sentiment Score: {pearson_corr_open}")
print(f"Pearson correlation between High and Sentiment Score: {pearson_corr_high}")
print(f"Pearson correlation between Low and Sentiment Score: {pearson_corr_low}")

print(f"Spearman correlation between Open and Sentiment Score: {spearman_corr_open}")
print(f"Spearman correlation between High and Sentiment Score: {spearman_corr_high}")
print(f"Spearman correlation between Low and Sentiment Score: {spearman_corr_low}")

print(f"Kendall correlation between Open and Sentiment Score: {kendall_corr_open}")
print(f"Kendall correlation between High and Sentiment Score: {kendall_corr_high}")
print(f"Kendall correlation between Low and Sentiment Score: {kendall_corr_low}")
