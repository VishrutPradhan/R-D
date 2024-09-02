import pandas as pd
import numpy as np
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer

def categorize_sentiment(compound_score):
    if compound_score >= 0.05:
        return 'Positive'
    elif compound_score <= -0.05:
        return 'Negative'
    else:
        return 'Neutral'

def analyze_sentiment(text):
    try:
        analyzer = SentimentIntensityAnalyzer()
        scores = analyzer.polarity_scores(text)
        return scores['compound']
    except:
        return np.nan  # Return NaN for any errors during text analysis

if __name__ == "__main__":
    # Load the CSV file with specified encoding
    df = pd.read_csv('DEC.csv', encoding='latin1')

    # Convert 'CreatedAt' column to datetime format, specifying the exact format
    date_format = '%a %b %d %H:%M:%S %z %Y'  # Adjusted to match your input data
    df['CreatedAt'] = pd.to_datetime(df['CreatedAt'], format=date_format, errors='coerce')

    # Apply sentiment analysis function
    df['SentimentScore'] = df['full_text'].apply(analyze_sentiment)
    df['SentimentCategory'] = df['SentimentScore'].apply(categorize_sentiment)

    # Calculate overall polarity score and sentiment for each date
    grouped = df.groupby(df['CreatedAt'].dt.date).agg({
        'SentimentScore': ['mean']
    })
    grouped.columns = ['DateOverallPolarity']
    grouped['DateSentiment'] = grouped['DateOverallPolarity'].apply(categorize_sentiment)

    # Join back to the original DataFrame
    df = df.join(grouped, on=df['CreatedAt'].dt.date)

    # Save the DataFrame to a new CSV file
    df.to_csv('sentiment_scores_output_Of_VADER.csv', index=False)

    print("Sentiment scores and categories saved to 'sentiment_scores_output.csv'")
