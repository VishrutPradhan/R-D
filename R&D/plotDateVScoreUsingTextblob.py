
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.dates import DateFormatter, DayLocator

# Load the data from the Excel file
df = pd.read_excel('dateAndScoreUsingTextblob.xlsx')
df['Date'] = pd.to_datetime(df['Date'])  # Ensure Date is in datetime format for better handling

# Filter data to only include December dates if necessary
df = df[(df['Date'] >= '2023-12-01') & (df['Date'] <= '2023-12-31')]

# Plotting
fig, ax1 = plt.subplots(figsize=(12, 6))

# Plot the stock prices
ax1.plot(df['Date'], df['Open'], label='Open', color='b')
ax1.plot(df['Date'], df['High'], label='High', color='g')
ax1.plot(df['Date'], df['Low'], label='Low', color='r')
ax1.set_xlabel('CreatedAt')
ax1.set_ylabel('Stock Prices')

# Set up date formatting and locators
date_formatter = DateFormatter('%Y-%m-%d')
day_locator = DayLocator()  # Locator that targets every day
ax1.xaxis.set_major_locator(day_locator)
ax1.xaxis.set_major_formatter(date_formatter)
ax1.tick_params(axis='x', rotation=90)  # Rotate labels to prevent overlap

ax1.legend(loc='upper left')

# Create a secondary y-axis for sentiment scores
ax2 = ax1.twinx()
ax2.plot(df['Date'], df['SentimentScore'], label='Sentiment Score', color='purple')
ax2.set_ylabel('Sentiment Score')
ax2.legend(loc='upper right')

# Set the y-axis limit for sentiment scores from -1 to 1
ax2.set_ylim(-1, 1)

# Add horizontal lines for neutral score thresholds at ±0.1
# ax2.axhline(0.1, color='gray', linestyle='--', linewidth=1)
ax2.axhline(0.1, color='gray', linestyle='--', linewidth=1)
ax2.axhline(0, color='gray', linestyle='--', linewidth=1)
ax2.axhline(-0.1, color='gray', linestyle='--', linewidth=1)

# Add vertical lines for each date in the DataFrame where data is available
for date in df['Date']:
    ax1.axvline(x=date, color='grey', linestyle='--', alpha=0.5)  # Light vertical lines for each date

plt.title('Stock Prices and Sentiment Score Correlation for December')
plt.tight_layout()

plt.show()
