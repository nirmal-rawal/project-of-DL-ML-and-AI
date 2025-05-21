import streamlit as st
import pickle
import numpy as np
import re
import nltk
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
import tensorflow as tf
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.layers import SpatialDropout1D, SimpleRNN, LSTM, Bidirectional
from tensorflow.keras.preprocessing.text import Tokenizer
import os

# Set TensorFlow logging to error only
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

# Verify all required model files exist
REQUIRED_MODEL_FILES = [
    'all_model/lstm_model.h5',
    'all_model/lstm_w2v_model.h5',
    'all_model/rnn_model.h5',
    'all_model/tokenizer.pickle'
]

for file_path in REQUIRED_MODEL_FILES:
    if not os.path.exists(file_path):
        st.error(f"Missing required file: {file_path}")
        st.error("Please ensure all model files exist in the 'all_model' folder")
        st.stop()

# Download NLTK resources
try:
    nltk.download('stopwords', quiet=True)
    nltk.download('wordnet', quiet=True)
except Exception as e:
    st.error(f"Failed to download NLTK resources: {str(e)}")
    st.stop()

# Constants
MAX_SEQUENCE_LENGTH = 200
EMBEDDING_DIM = 300

# Initialize lemmatizer and stopwords
lemmatizer = WordNetLemmatizer()
stop_words = set(stopwords.words('english'))

# Text cleaning function
def clean_text(text):
    # Remove URLs
    text = re.sub(r'http\S+|www\S+|https\S+', '', text, flags=re.MULTILINE)
    # Remove mentions and hashtags
    text = re.sub(r'\@\w+|\#', '', text)
    # Remove special characters and numbers
    text = re.sub(r'[^a-zA-Z\s]', '', text)
    # Convert to lowercase
    text = text.lower()
    # Tokenize
    tokens = text.split()
    # Remove stopwords and lemmatize
    tokens = [lemmatizer.lemmatize(word) for word in tokens if word not in stop_words]
    return ' '.join(tokens)

# Custom objects for model loading with TensorFlow 2.12 compatibility
custom_objects = {
    'SpatialDropout1D': SpatialDropout1D,
    'SimpleRNN': SimpleRNN,
    'LSTM': LSTM,
    'Bidirectional': Bidirectional,
}

# Load models and tokenizer with comprehensive error handling
@st.cache_resource
def load_models():
    try:
        # Load tokenizer
        with open('all_model/tokenizer.pickle', 'rb') as handle:
            tokenizer = pickle.load(handle)
            
        # Verify the tokenizer has required methods
        if not hasattr(tokenizer, 'texts_to_sequences'):
            raise AttributeError("Loaded tokenizer is invalid")
            
    except Exception as e:
        st.error(f"Failed to load tokenizer: {str(e)}")
        st.error("Please ensure the tokenizer.pickle file exists and is valid")
        raise
    
    try:
        # Load models with custom objects
        rnn_model = load_model('all_model/rnn_model.h5', custom_objects=custom_objects)
        lstm_model = load_model('all_model/lstm_model.h5', custom_objects=custom_objects)
        lstm_w2v_model = load_model('all_model/lstm_w2v_model.h5', custom_objects=custom_objects)
        
    except Exception as e:
        st.error(f"Failed to load models: {str(e)}")
        st.error("""
        Possible solutions:
        1. Ensure all model files exist in the 'all_model' folder
        2. Make sure you're using TensorFlow 2.12.0
        3. Try reinstalling requirements: pip install tensorflow==2.12.0 numpy nltk streamlit
        """)
        raise
    
    return tokenizer, rnn_model, lstm_model, lstm_w2v_model

# Prediction function
def predict_news(text, model, tokenizer, model_name):
    try:
        # Clean the text
        cleaned_text = clean_text(text)
        
        # Tokenize and pad the sequence
        sequence = tokenizer.texts_to_sequences([cleaned_text])
        if not sequence or not sequence[0]:  # Handle empty sequences
            return {
                "model": model_name,
                "label": "Unknown",
                "confidence": 0.5,
                "interpretation": f"The {model_name} couldn't process this text (no valid tokens)"
            }
            
        padded_sequence = pad_sequences(sequence, maxlen=MAX_SEQUENCE_LENGTH, padding='post', truncating='post')
        
        # Make prediction
        prediction = model.predict(padded_sequence)
        confidence = float(prediction[0][0])
        
        if confidence > 0.5:
            return {
                "model": model_name,
                "label": "True",
                "confidence": confidence,
                "interpretation": f"The {model_name} predicts this is TRUE news with {confidence*100:.2f}% confidence"
            }
        else:
            return {
                "model": model_name,
                "label": "Fake",
                "confidence": 1 - confidence,
                "interpretation": f"The {model_name} predicts this is FAKE news with {(1-confidence)*100:.2f}% confidence"
            }
    except Exception as e:
        st.error(f"Prediction failed in {model_name}: {str(e)}")
        return {
            "model": model_name,
            "label": "Error",
            "confidence": 0.0,
            "interpretation": f"The {model_name} encountered an error processing this text"
        }

# Main Streamlit app
def main():
    st.title("Fake News Detection System")
    st.write("""
    This app uses deep learning models to classify news articles as either "True" or "Fake".
    """)
    
    # Display requirements warning
    st.sidebar.warning("""
    **Requirements:**
    - Python 3.7+
    - TensorFlow 2.12.0
    - Run: `pip install tensorflow==2.12.0 numpy nltk streamlit`
    """)
    
    try:
        # Load models
        tokenizer, rnn_model, lstm_model, lstm_w2v_model = load_models()
    except Exception as e:
        st.error(f"Failed to initialize the app: {str(e)}")
        st.stop()
    
    # Input text
    news_text = st.text_area("Enter the news article text:", height=200, 
                           placeholder="Paste news article content here...")
    
    # Model selection
    selected_models = st.multiselect(
        "Select models to use:",
        ["Simple RNN", "LSTM", "LSTM with Word2Vec"],
        default=["LSTM with Word2Vec"]
    )
    
    if st.button("Analyze"):
        if not news_text.strip():
            st.warning("Please enter some text to analyze.")
            return
            
        results = []
        
        try:
            if "Simple RNN" in selected_models:
                results.append(predict_news(news_text, rnn_model, tokenizer, "Simple RNN"))
            
            if "LSTM" in selected_models:
                results.append(predict_news(news_text, lstm_model, tokenizer, "LSTM"))
            
            if "LSTM with Word2Vec" in selected_models:
                results.append(predict_news(news_text, lstm_w2v_model, tokenizer, "LSTM with Word2Vec"))
            
            # Display results
            st.subheader("Analysis Results")
            
            for result in results:
                if result["label"] == "Error":
                    color = "gray"
                elif result["label"] == "True":
                    color = "green"
                elif result["label"] == "Fake":
                    color = "red"
                else:
                    color = "orange"
                
                with st.expander(f"{result['model']} - Prediction: :{color}[{result['label']}]"):
                    st.write(result["interpretation"])
                    
                    if result["label"] not in ["Error", "Unknown"]:
                        st.progress(result["confidence"] if result["label"] == "True" else (1 - result["confidence"]))
                        st.markdown("---")
                        col1, col2 = st.columns(2)
                        with col1:
                            st.metric("Confidence in True", f"{result['confidence']*100:.2f}%")
                        with col2:
                            st.metric("Confidence in Fake", f"{(1-result['confidence'])*100:.2f}%")
        except Exception as e:
            st.error(f"Analysis failed: {str(e)}")

if __name__ == "__main__":
    main()