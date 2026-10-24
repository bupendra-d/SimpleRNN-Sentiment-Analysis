import streamlit as st
import tensorflow as tf

from tensorflow.keras.datasets import imdb
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.preprocessing.text import text_to_word_sequence

# Configuration
MAX_FEATURES = 10_000
MAX_LEN = 500

st.set_page_config(page_title="IMDB Sentiment Analysis", page_icon="🎬")

@st.cache_resource
def load_sentiment_model():
    return tf.keras.models.load_model("simple_rnn_imdb.keras")

@st.cache_data
def load_word_index():
    return imdb.get_word_index()

model = load_sentiment_model()
word_index = load_word_index()


def preprocess_text(text):
    words = text_to_word_sequence(text)
    encoded_review = []

    for word in words:
        index = word_index.get(word)

        if index is None or index + 3 >= MAX_FEATURES:
            encoded_review.append(2)  # Out-of-vocabulary token
        else:
            encoded_review.append(index + 3)

    padded_review = pad_sequences(
        [encoded_review],
        maxlen=MAX_LEN
    )

    return padded_review


# Streamlit interface
st.title("IMDB Movie Review Sentiment Analysis")

st.write(
    "Enter a movie review to predict whether its sentiment is positive or negative."
)

review = st.text_area("Enter your movie review")

if st.button("Analyze Sentiment"):
    if not review.strip():
        st.warning("Please enter a movie review.")
    else:
        processed_review = preprocess_text(review)
        score = float(model.predict(processed_review, verbose=0)[0][0])

        if score >= 0.5:
            st.success("Sentiment: Positive")
        else:
            st.error("Sentiment: Negative")

        st.metric("Model Score", f"{score:.2%}")