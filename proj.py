import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
from wordcloud import WordCloud
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.decomposition import TruncatedSVD
from transformers import BertTokenizer, BertForSequenceClassification
import torch

# Load datasets
@st.cache_data
def load_data(file):
    return pd.read_csv(file)

@st.cache_data
def load_words(file):
    if file is not None:
        content = file.read().decode('utf-8')
        words = content.splitlines()
        return words
    return []
# Text classification function
def predict_sentiment(text, positive_words, negative_words):
    # Use positive and negative word lists for prediction
    positive_count = sum(1 for word in text.lower().split() if word in positive_words)
    negative_count = sum(1 for word in text.lower().split() if word in negative_words)

    # Simple rule-based sentiment analysis
    if positive_count > negative_count:
        return "Not in Depression"
    else:
        return "Depression"

# EDA function
def eda(df):
    # Add your exploratory data analysis code here (using plotly)
    # Example: Visualization of retweets over time
    fig = px.line(df, x='date', y='retweets', title='Retweets Over Time')
    st.plotly_chart(fig)

# Data cleaning function
def clean_data(df):
    # Add your data cleaning code here
    # Example: Removing duplicates
    df = df.drop_duplicates()
    return df

# Wordcloud function
def generate_wordcloud(text, title):
    # Add your wordcloud generation code here
    wordcloud = WordCloud(width=800, height=400, background_color='white').generate(text)
    st.title(title)
    st.image(wordcloud.to_array(), use_container_width=True)

# Dimension Reduction function
def dimension_reduction(text_data):
    # Add your dimension reduction code here
    # Example: Using Truncated SVD
    vectorizer = TfidfVectorizer(stop_words='english')
    X = vectorizer.fit_transform(text_data)
    svd = TruncatedSVD(n_components=2)
    X_reduced = svd.fit_transform(X)

    # Visualize the dimension-reduced data
    fig = px.scatter(x=X_reduced[:, 0], y=X_reduced[:, 1], labels={'x': 'Dimension 1', 'y': 'Dimension 2'},
                     title='Dimension Reduction')
    st.plotly_chart(fig)

# Streamlit app
def main():
    st.title("Depression Detection in Social Media")

    # Upload datasets
    st.sidebar.header("Upload Datasets")
    data_file = st.sidebar.file_uploader("Upload Comments Dataset (CSV)", type=["csv"])
    positive_words_file = st.sidebar.file_uploader("Upload Positive Words (TXT)", type=["txt"])
    negative_words_file = st.sidebar.file_uploader("Upload Negative Words (TXT)", type=["txt"])

    if data_file is not None and positive_words_file is not None and negative_words_file is not None:
        # Load data
        comments_df = load_data(data_file)
        positive_words = load_words(positive_words_file)
        negative_words = load_words(negative_words_file)

        # Data cleaning
        comments_df = clean_data(comments_df)

        # Exploratory Data Analysis
        st.sidebar.header("Exploratory Data Analysis")
        if st.sidebar.checkbox("Show EDA"):
            eda(comments_df)

        # Wordclouds
        st.sidebar.header("Word Clouds")
        if st.sidebar.checkbox("Show Word Clouds"):
            text_all = " ".join(comment for comment in comments_df['text'])
            generate_wordcloud(text_all, "Word Cloud - All Comments")

        # Dimension Reduction
        st.sidebar.header("Dimension Reduction")
        if st.sidebar.checkbox("Show Dimension Reduction"):
            text_data = comments_df['text'].tolist()
            dimension_reduction(text_data)

        # Model prediction
        st.header("Model Prediction")

        # User-based input
        user_input = st.text_area("Enter Text for Prediction:")
        if st.button("Predict"):
            user_prediction = predict_sentiment(user_input, positive_words, negative_words)
            st.write(f"User Input Prediction: {user_prediction}")

        # Dataset-based input
        st.sidebar.header("Dataset-based Input")
        selected_comment = st.sidebar.selectbox("Select Comment for Prediction:", comments_df['text'])
        if st.sidebar.button("Predict Comment"):
            dataset_prediction = predict_sentiment(selected_comment, positive_words, negative_words)
            st.sidebar.write(f"Dataset Comment Prediction: {dataset_prediction}")

if __name__ == "__main__":
    main()
