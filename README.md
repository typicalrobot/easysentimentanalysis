# easysentimentanalysis

A simple, straightforward approach to sentiment analysis in Python using pandas, TextBlob, and optional advanced models from Hugging Face Transformers. This repository shows how to read in data (for example, a CSV of YouTube comments), clean and preprocess text, and then derive sentiment scores or labels (Positive, Negative, Neutral). <br>

## Table of Contents <br>
Features

Requirements

Usage

Running in Google Colab

Contributing

<br>

## Features <br>
CSV reading with pandas: Quickly load and inspect datasets.

Text cleaning: Remove URLs, punctuation, and convert text to lowercase.

Sentiment Analysis using:

TextBlob (simple rule-based approach).

Hugging Face Transformers (optional, more advanced approach).

<br>

## Requirements <br>

Python 3.7+

pandas

TextBlob

Matplotlib (optional, for charts)

transformers and torch (optional, if using advanced models)

<br>
You can install everything via a requirements.txt (if provided) or manually with pip:

pip install pandas textblob matplotlib transformers torch
<br>

## Usage <br>
The core workflow is generally:

Load your CSV file into a pandas DataFrame.

Clean or preprocess the text (remove URLs, punctuation, etc.).

Apply sentiment analysis with TextBlob or a Transformer-based model.

Review or visualize the results.

Below is a minimal sample script (assuming your file is named data.csv and contains a column called Comment Text):

import pandas as pd
import re
from textblob import TextBlob

### 1. Load data
df = pd.read_csv("data.csv")

### 2. Clean text
def clean_text(text):
    text = text.lower()
    text = re.sub(r'https?://\S+|www\.\S+', '', text)  # remove URLs
    text = re.sub(r'[^a-zA-Z0-9\s]', '', text)         # remove punctuation/special chars
    return text.strip()

df['CleanText'] = df['Comment Text'].apply(clean_text)

### 3. Apply TextBlob sentiment analysis
def get_sentiment_textblob(text):
    return TextBlob(text).sentiment.polarity  # [-1.0, 1.0]

df['Polarity'] = df['CleanText'].apply(get_sentiment_textblob)

### 4. Convert polarity to categories
def categorize_polarity(p):
    if p > 0:
        return "Positive"
    elif p < 0:
        return "Negative"
    else:
        return "Neutral"

df['Sentiment'] = df['Polarity'].apply(categorize_polarity)

### 5. Inspect or visualize
print(df[['Comment Text', 'Polarity', 'Sentiment']].head())
<br>
Running in Google Colab <br>
Upload your dataset (e.g., data.csv) to Colab’s file system.

Copy and paste the example code (or your own script) into a Colab cell.

Install needed libraries within Colab:
!pip install pandas textblob matplotlib torch transformers

Run the cells to produce sentiment results. You can add a bar chart, for example:
import matplotlib.pyplot as plt

df['Sentiment'].value_counts().plot(kind='bar')
plt.title('Sentiment Distribution')
plt.xlabel('Sentiment')
plt.ylabel('Count')
plt.show()
<br>
## Contributing <br>
Contributions are welcome! Feel free to:

Submit issues for bugs or feature requests.

Fork the repo and open a pull request with new ideas or improvements.

<br>
