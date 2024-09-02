from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score
import pandas as pd

# Your data
data = {
    'Date': ['29-12-2023', '28-12-2023', '27-12-2023', '26-12-2023', '22-12-2023', '21-12-2023', '20-12-2023', '19-12-2023', '18-12-2023', '15-12-2023', '14-12-2023', '13-12-2023', '12-12-2023', '11-12-2023', '08-12-2023', '07-12-2023', '06-12-2023', '05-12-2023', '04-12-2023', '01-12-2023'],
    'Open': [634.1, 639, 635, 643.9, 646, 615, 627.9, 623.95, 610, 600, 605, 616.55, 655, 664.9, 669.7, 728.85, 845, 860, 878.35, 875.85],
    'High': [643, 640.5, 641.95, 644.9, 650.6, 641.05, 635.95, 626, 624, 609.45, 616.4, 616.75, 658, 667, 679.85, 728.85, 847, 867, 881.15, 884.15],
    'Low': [631.35, 631.15, 630.15, 630, 635.1, 606.05, 610, 618.3, 607, 599.05, 603.2, 592.7, 614.8, 651.2, 637, 650.45, 810.05, 837, 853.15, 867.2],
    'Sentiment': ['Neutral', 'Positive', 'Positive', 'Neutral', 'Positive', 'Positive', 'Neutral', 'Neutral', 'Neutral', 'Neutral', 'Neutral', 'Neutral', 'Neutral', 'Neutral', 'Neutral', 'Neutral', 'Positive', 'Positive', 'Positive', 'Positive'],
    'SentimentScore': [0.099726686, 0.140969518, 0.109632387, 0.083223249, 0.100586355, 0.120252741, 0.040800903, 0.039489122, 0.031248197, -0.001318436, 0.089623864, -0.052592593, 0.078489583, 0.048585859, 0.048961327, 0.080987314, 0.154762098, 0.128178845, 0.143720256, 0.114615686]
}

df = pd.DataFrame(data)

# Convert 'Sentiment' to numerical
df['Sentiment'] = df['Sentiment'].map({'Neutral': 0, 'Positive': 1})

# Predict next day's Open, High, Low based on SentimentScore
X = df['SentimentScore'].values.reshape(-1, 1)
y_open = df['Open'].shift(-1).dropna()
y_high = df['High'].shift(-1).dropna()
y_low = df['Low'].shift(-1).dropna()

# Split the data into training and testing sets
X_train, X_test, y_train_open, y_test_open, y_train_high, y_test_high, y_train_low, y_test_low = train_test_split(X, y_open, y_high, y_low, test_size=0.2, random_state=0)

# Build the Linear Regression models
model_open = LinearRegression()
model_high = LinearRegression()
model_low = LinearRegression()
model_open.fit(X_train, y_train_open)
model_high.fit(X_train, y_train_high)
model_low.fit(X_train, y_train_low)

# Make predictions
y_pred_open = model_open.predict(X_test)
y_pred_high = model_high.predict(X_test)
y_pred_low = model_low.predict(X_test)

# Evaluate the models
mse_open = mean_squared_error(y_test_open, y_pred_open)
r2_open = r2_score(y_test_open, y_pred_open)
mse_high = mean_squared_error(y_test_high, y_pred_high)
r2_high = r2_score(y_test_high, y_pred_high)
mse_low = mean_squared_error(y_test_low, y_pred_low)
r2_low = r2_score(y_test_low, y_pred_low)

print("Open Price Prediction:")
print(f"Mean Squared Error: {mse_open}")
print(f"R-squared: {r2_open}")

print("\nHigh Price Prediction:")
print(f"Mean Squared Error: {mse_high}")
print(f"R-squared: {r2_high}")

print("\nLow Price Prediction:")
print(f"Mean Squared Error: {mse_low}")
print(f"R-squared: {r2_low}")
