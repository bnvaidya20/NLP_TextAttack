# Import libraries
import pandas as pd
import re
import unidecode

from bs4 import BeautifulSoup
import pickle
from scipy.sparse import save_npz

from sklearn.datasets import fetch_20newsgroups

from sklearn.feature_extraction.text import CountVectorizer


# Data Fetching and Basic Cleaning
def fetch_and_clean_data(categories, subset='train'):
    data = fetch_20newsgroups(subset=subset, categories=categories, shuffle=True, 
                              remove=('headers', 'footers', 'quotes'), random_state=42)
    
    return [clean_text(text) for text in data.data], data.target

# Text Cleaning Function
def clean_text(text, max_len=300):
    soup = BeautifulSoup(text, 'html.parser')
    clean_text = re.sub('<.*?>', ' ', soup.text)  
    clean_text = unidecode.unidecode(clean_text)  
    clean_text = re.sub('[^A-Za-z0-9.]+', ' ', clean_text).lower()  
    return clean_text[:max_len]

# Save data
def save_pkl_file(filepath, data):
    with open(filepath, 'wb') as file:
        pickle.dump(data, file)

# Load the data from files
def load_pkl_file(filepath):
    with open(filepath, 'rb') as file:
        data = pickle.load(file)
    return data

# Feature Extraction
def extract_features(data, save_path=None, suffix=None):
    """
    Extracts features from text data and optionally saves the feature matrix with a suffix.
    """
    vectorizer = CountVectorizer()
    features = vectorizer.fit_transform(data)
       
    if save_path:
        # Append the suffix before .npz extension
        if suffix and not save_path.endswith(suffix + '.npz'):
            save_path = save_path.replace('.npz', '') + f'_{suffix}.npz'
        
        save_npz(save_path, features)
        print(f"Feature matrix saved to {save_path}")
    
    return features

def convert_to_dataframe(results):
    # Convert results to a DataFrame 
    df = pd.DataFrame(results)

    return df