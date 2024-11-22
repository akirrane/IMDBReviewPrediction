# Import libraries
import numpy as np
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer
import re

# Initialize the lemmatizer and stopwords
lemmatizer = WordNetLemmatizer()
stop_words = set(stopwords.words('english'))

# Clean and tokenize function
def clean_tokenize(sentence):
    """
    Clean, tokenize, and lemmatize a sentence, removing stopwords and punctuation.
    """
    # Lowercase the sentence
    sentence = sentence.lower()
    
    # Remove unwanted characters (e.g., special characters, extra spaces)
    sentence = re.sub(r"[^a-zA-Z\s]", '', sentence)  # Keep only letters and spaces
    
    # Tokenize
    words = word_tokenize(sentence)
    
    # Remove stop words and lemmatize each word
    words = [lemmatizer.lemmatize(word) for word in words if word not in stop_words]
    
    return words

# Generates a bag-of-words vector
def bag_of_words(tokenized_sentence, words):
    """
    Create a bag-of-words representation: a binary array of 0s and 1s indicating the presence
    of each word in the vocabulary.
    """
    # Lemmatize each word in the tokenized sentence
    sentence_words = [lemmatizer.lemmatize(word.lower()) for word in tokenized_sentence]
    bag = np.zeros(len(words), dtype=np.float32)  # Initialize a bag of zeros for each word
    
    # For each word in the vocabulary
    for idx, w in enumerate(words):
        # If the word exists in the lemmatized tokenized sentence, set to 1
        if w in sentence_words: 
            bag[idx] = 1

    return bag

# Optional function: Sentence embedding (useful if switching to pretrained embeddings)
def sentence_to_embedding(sentence, vocab, embedding_dim=100, max_len=50):
    """
    Convert a sentence into a fixed-size embedding vector using pretrained word embeddings (e.g., GloVe).
    If the word is not in the vocab, use a zero vector. Pads or truncates to max_len words.
    """
    # Clean and tokenize the sentence
    words = clean_tokenize(sentence)
    
    # Convert each word into its embedding vector or a zero vector if not in vocab
    embeddings = [vocab[word].numpy() if word in vocab else np.zeros(embedding_dim) for word in words]
    
    # Pad or truncate to fixed length
    embeddings = embeddings[:max_len] if len(embeddings) >= max_len else embeddings + [np.zeros(embedding_dim)] * (max_len - len(embeddings))
    
    # Flatten the list of embeddings into a single vector
    return np.array(embeddings).flatten()