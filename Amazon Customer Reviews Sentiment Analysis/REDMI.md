## Project Overview
This project analyzes customer reviews from Amazon to classify them as positive or negative sentiment using Natural Language Processing (NLP) techniques and machine learning models.


## Dataset
The dataset contains Amazon customer reviews with:
- Review text
- Product ratings (1-5 stars)
- Sentiment labels (positive/negative)
- Approximately 10,000 reviews with class distribution:
  - Positive (4-5 stars): 84.8%
  - Negative (1-2 stars): 15.2%

## Preprocessing Steps
1. **Text Cleaning**:
   - Convert to lowercase
   - Remove special characters
   - Remove stopwords
   - Lemmatization

2. **Feature Engineering**:
   - Bag-of-Words (BoW)
   - TF-IDF
   - N-grams (unigrams to 4-grams)

3. **Visualization**:
   - Word clouds
   - Review length distribution
   - Sentiment class distribution

## Models Implemented

### 1. Logistic Regression with Bag-of-Words
- **Features**: Unigrams
- **Performance**: F1-score = 0.956

### 2. Logistic Regression with N-grams (1-4)
- **Features**: Unigrams to 4-grams
- **Performance**: F1-score = 0.956

### 3. TF-IDF Model
- **Features**: TF-IDF weighted unigrams
- **Performance**: (To be evaluated)

## Key Findings
- Most important positive words: "great", "delicious", "good", "the best", "perfect"
- Most important negative words: "disappointed", "not", "bad", "worst", "waste"
- Baseline accuracy (all positive): 84.8%
- Best model achieves ~95.6% F1-score


