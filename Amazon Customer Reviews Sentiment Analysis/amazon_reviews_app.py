import streamlit as st
import pandas as pd
import numpy as np
import joblib
import pickle
import re
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
import nltk
import warnings

# Suppress warnings
warnings.filterwarnings("ignore", category=UserWarning)

# Download all required NLTK resources
nltk.download('punkt', quiet=True)
nltk.download('stopwords', quiet=True)
nltk.download('wordnet', quiet=True)
nltk.download('omw-1.4', quiet=True)

# Set page config
st.set_page_config(
    page_title="Amazon Reviews Sentiment Analysis",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Load models and vectorizers
@st.cache_resource
def load_models():
    try:
        # Load logistic regression model
        lr_model = joblib.load('models/logistic_regression_model.pkl')
        
        # Load vectorizers - make sure these match what the model was trained on
        with open('models/bow_vectorizer.pkl', 'rb') as f:
            bow_vectorizer = pickle.load(f)
        
        with open('models/tfidf_vectorizer.pkl', 'rb') as f:
            tfidf_vectorizer = pickle.load(f)
        
        # Verify dimensions match
        test_vec = bow_vectorizer.transform(["test review"])
        if test_vec.shape[1] != lr_model.coef_.shape[1]:
            raise ValueError(
                f"Dimension mismatch! Model expects {lr_model.coef_.shape[1]} features, "
                f"but vectorizer produces {test_vec.shape[1]} features."
            )
            
        return lr_model, bow_vectorizer, tfidf_vectorizer
    except Exception as e:
        st.error(f"Error loading models: {str(e)}")
        raise e

try:
    lr_model, bow_vectorizer, tfidf_vectorizer = load_models()
except Exception as e:
    st.error("Failed to load models. Please check the model files.")
    st.stop()

# Preprocessing function
def preprocess_text(text):
    if not isinstance(text, str):
        return ""
        
    # Convert to lowercase
    text = text.lower()
    
    # Remove special characters
    text = re.sub(r'[^a-zA-Z0-9\s]', ' ', text)
    
    # Tokenize and remove stopwords
    stop_words = set(stopwords.words('english'))
    words = word_tokenize(text)
    words = [word for word in words if word not in stop_words and word.isalpha()]
    
    return ' '.join(words)

# Sentiment analysis function with dimension verification
def analyze_sentiment(text, vectorizer, model):
    try:
        # Preprocess the text
        processed_text = preprocess_text(text)
        
        # Vectorize the text
        text_vector = vectorizer.transform([processed_text])
        
        # Verify dimensions
        if text_vector.shape[1] != model.coef_.shape[1]:
            raise ValueError(
                f"Feature dimension mismatch. "
                f"Vectorizer produced {text_vector.shape[1]} features, "
                f"but model expects {model.coef_.shape[1]} features."
            )
        
        # Predict sentiment
        prediction = model.predict(text_vector)
        probability = model.predict_proba(text_vector)
        
        return prediction[0], probability[0]
    except Exception as e:
        st.error(f"Error in sentiment analysis: {str(e)}")
        raise e

# Main app
def main():
    st.title("Amazon Reviews Sentiment Analysis")
    st.write("""
    This app analyzes the sentiment of Amazon product reviews using Natural Language Processing (NLP) 
    and machine learning. Enter a review in the text box below to see if it's positive or negative.
    """)
    
    # Create tabs
    tab1, tab2, tab3 = st.tabs(["Analyze Review", "Model Info", "About"])
    
    with tab1:
        st.header("Review Analysis")
        
        # Text input
        review_text = st.text_area("Enter your Amazon product review:", 
                                 "My cats have been happily eating Felidae Platinum for more than two years. I just got a new bag and the shape of the food is different. They tried the new food when I first put it in their bowls and now the bowls sit full and the kitties will not touch the food. I've noticed similar reviews related to formula changes in the past. Unfortunately, I now need to find a new food that my cats will eat. ")
        
        # Model selection
        model_option = st.radio("Select vectorization method:",
                              ("Bag-of-Words", "TF-IDF"))
        
        if st.button("Analyze Sentiment"):
            if review_text.strip() == "":
                st.warning("Please enter a review to analyze.")
            else:
                with st.spinner("Analyzing sentiment..."):
                    try:
                        if model_option == "Bag-of-Words":
                            vectorizer = bow_vectorizer
                        else:
                            vectorizer = tfidf_vectorizer
                        
                        prediction, probability = analyze_sentiment(review_text, vectorizer, lr_model)
                        
                        # Display results
                        st.subheader("Results")
                        
                        col1, col2 = st.columns(2)
                        
                        with col1:
                            st.metric("Sentiment", 
                                     "Positive" if prediction == 1 else "Negative",
                                     delta=f"{probability[prediction]*100:.2f}% confidence")
                        
                        with col2:
                            # Progress bars for probabilities
                            st.write("Confidence levels:")
                            st.progress(probability[1], text=f"Positive: {probability[1]*100:.2f}%")
                            st.progress(probability[0], text=f"Negative: {probability[0]*100:.2f}%")
                        
                        # Display interpretation
                        if prediction == 1:
                            st.success("This review expresses a positive sentiment.")
                        else:
                            st.error("This review expresses a negative sentiment.")
                    except Exception as e:
                        st.error(f"An error occurred during analysis: {str(e)}")

    with tab2:
        st.header("Model Information")
        st.write("""
        ### Logistic Regression Model
        This app uses a logistic regression model trained on Amazon product reviews to classify sentiment.
        
        **Model Performance:**
        - F1 Score: 0.956 (Bag-of-Words)
        - F1 Score: 0.939 (TF-IDF)
        
        **Features:**
        - Text preprocessing: Lowercasing, special character removal, stopword removal
        - Two vectorization methods available:
            - Bag-of-Words (BoW)
            - TF-IDF (Term Frequency-Inverse Document Frequency)
        """)
        
        st.subheader("Top Predictive Features")
        
        # Show top features for positive and negative sentiment
        col1, col2 = st.columns(2)
        
        with col1:
            st.write("**Positive Sentiment Indicators**")
            st.write("""
            - love
            - great
            - excellent
            - perfect
            - amazing
            - wonderful
            - best
            - delicious
            - happy
            - fantastic
            """)
        
        with col2:
            st.write("**Negative Sentiment Indicators**")
            st.write("""
            - worst
            - bad
            - terrible
            - awful
            - horrible
            - disappointed
            - waste
            - broken
            - poor
            - return
            """)
    
    with tab3:
        st.header("About This App")
        st.write("""
        This sentiment analysis application was developed to demonstrate how machine learning 
        can be used to analyze customer reviews from Amazon.
        
        **Key Features:**
        - Real-time sentiment analysis of product reviews
        - Two different NLP feature extraction methods
        - Confidence level visualization
        
        **Technical Details:**
        - Built with Python and Streamlit
        - Uses NLTK for text preprocessing
        - Logistic Regression for classification
        
        **Dataset:**
        - 10,000 Amazon product reviews
        - Balanced for positive and negative sentiment
        """)
        
        st.write("Developed by [Your Name]")

if __name__ == "__main__":
    main()