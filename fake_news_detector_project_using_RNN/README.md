# Fake News Detection with RNNs and LSTMs

**Student ID:** 2358113  
**Name:** Nirmal Rawal

This project implements various neural network architectures (Simple RNN, LSTM, and LSTM with Word2Vec embeddings) for classifying news articles as "True" or "Fake".

  ![Screenshot 2025-05-16 212517](https://github.com/user-attachments/assets/37348878-517a-4863-849d-9d5b6b8c46ad)
  ![Screenshot 2025-05-16 212822](https://github.com/user-attachments/assets/9cde4601-2075-47a5-b3af-cdd05365e101)


## Dataset

The dataset contains news articles labeled as:
- True (1): 10,000+ samples
- Fake (0): 10,000+ samples

## Preprocessing

1. Text cleaning:
   - URL removal
   - Mention/hashtag removal
   - Special character removal
   - Lowercasing
   - Lemmatization
   - Stopword removal

2. Tokenization and padding:
   - Max vocabulary size: 100,000 words
   - Max sequence length: 200 tokens

## Model Architectures

### 1. Simple RNN
- Embedding layer (300D)
- Spatial Dropout (0.2)
- SimpleRNN layer (100 units)
- Sigmoid output
- **Accuracy:** 93.3%

### 2. LSTM with Regularization
- Bidirectional LSTM (64 units)
- Increased dropout (0.3-0.4)
- L2 regularization
- **Accuracy:** 98.6%

### 3. LSTM with Word2Vec
- Pre-trained Word2Vec embeddings (Google News)
- Bidirectional LSTM (100 units)
- Frozen embedding layer
- **Accuracy:** 99.5%

## Training Details

- **Optimizer:** Adam with learning rate 1e-3 to 1e-4
- **Batch Size:** 32-64
- **Epochs:** 5-10 (with early stopping)
- **Validation Split:** 20%

## Results

| Model               | Accuracy | Validation Accuracy | Training Time per Epoch |
|---------------------|----------|---------------------|-------------------------|
| Simple RNN          | 93.3%    | 91.6%               | ~205s                   |
| LSTM                | 98.6%    | 98.8%               | ~630s                   |
| LSTM with Word2Vec  | 99.5%    | 99.6%               | ~440s                   |
