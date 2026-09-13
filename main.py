import numpy as np
import tensorflow as tf
from tensorflow.keras.datasets import imdb
from tensorflow.keras.preprocessing import sequence
from tensorflow.keras.models import load_model

##Load the imdb word index
word_index = imdb.get_word_index()
reverse_word_index = {value : key for key, value in word_index.items()}

#Load the pre-trained model with ReLu activation function
model = load_model('RNN_imdb.keras')

#Step 2 helper functions
#Function to decode reviews
def decode_review(encoded_review):
    return ' '.join([reverse_word_index.get(i-3, '?') for i in encoded_review])

#Function to interpret review
def preprocess_text(text):
    words = text.lower().split()
    encoded_review = [word_index.get(word,2) + 3 for word in words]
    padded_review = sequence.pad_sequences([encoded_review], maxlen = 500)
    return padded_review

## Prediction Function
# PREDICTIONN FUNCTION


def predict_sentiment(review):
    preprocessed_input = preprocess_text(review)

    prediction = model.predict(preprocessed_input)

    sentiment = "In support" if prediction[0][0]> 0.5 else "Unsupportive"

    return sentiment, prediction[0][0]


##Streamlit App
import streamlit as st

st.set_page_config(page_title="IMDB Sentiment Analysis", page_icon="🎬")

# st.title("🎬 IMDB Movie Review Sentiment Analysis")
# st.caption("Enter a movie review to classify it as positive or negative")

# user_input = st.text_area("Movie Review", placeholder="e.g. This movie was absolutely fantastic!")

# #User_input
# #user_input = st.text_area('Movie Review')

# if st.button('Classify'):
#     preprocessed_input = preprocess_text(user_input)

#     ## Make prediction
#     prediction = model.predict(preprocessed_input)
#     sentiment = 'Poositive' if prediction[0][0] >0.5 else 'Negative'

#     ## Display the result 
#     st.write(f'Sentiment : {sentiment}')
#     st.write(f'Prediction Score:{prediction[0][0]}')
# else:
#     st.write('Please enter a movie review')

st.set_page_config(page_title="IMDB Sentiment Analysis", page_icon="🎬")

st.title("🎬 IMDB Movie Review Sentiment Analysis")
st.caption("Enter a movie review to classify it as positive or negative")

review = st.text_area("Movie Review", placeholder="e.g. This movie was absolutely fantastic!")

if st.button("Classify"):
    if not review.strip():
        st.warning("Please enter a review first.")
    else:
        with st.spinner("Analyzing..."):
            sentiment, score = predict_sentiment(review)  # keep however yours actually returns these

        st.divider()
        if "pos" in sentiment.lower() or "support" in sentiment.lower():
            st.success(f"Sentiment: {sentiment}")
        else:
            st.error(f"Sentiment: {sentiment}")
        st.metric("Confidence", f"{score:.1%}")