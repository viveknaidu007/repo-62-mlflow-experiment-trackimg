import pandas as pd
import numpy as np

import re
import nltk
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer

import os
import warnings

import sklearn
from sklearn.model_selection import train_test_split

from sklearn.naive_bayes import GaussianNB

from sklearn.pipeline import Pipeline
from sklearn import metrics
from sklearn.naive_bayes import MultinomialNB
warnings.filterwarnings("ignore")

from prefect import task, flow

stopwords = set(stopwords.words('english'))
lemmatizer = WordNetLemmatizer()

@task
def load_data(file_path):
    """
    Load data from a CSV file.
    """
    return pd.read_csv(file_path)

@task
def split_inputs_output(data, inputs, output):
    """
    Split features and target variables.
    """
    x = data[inputs]
    y = data[output]
    return x, y

@task
def split_train_test(x, y, test_size=0.20, random_state=1):
    """
    Split data into train and test sets.
    """
    return train_test_split(x, y, test_size=test_size, random_state=random_state)

@task
def preprocess_text(text):
    """
    Preprocesses the input text by removing special characters, converting to lowercase, 
    tokenizing, removing stop words, and lemmatizing.

    Parameters:
    text (str): Input text to be preprocessed.

    Returns:
    str: Preprocessed text.
    """
    # Define custom stop words
    custom_stop_words = {'Hii', 'it', 'Product', 'Shuttle', 'hii', 'flipkart', 'flipkartread', 'product', 'productread', 'read', 'goodread','shuttle', 'Readmore'}

    
    # Remove special characters
    text = re.sub("[^a-zA-Z]", " ", text)
    
    # Convert words to lowercase
    text = text.lower()
    
    # Tokenization
    words = text.split()
    
    # Remove stop words
    words = [word for word in words if word not in stopwords.words('english') and word not in custom_stop_words]
    
    # Lemmatization
    words = [Lemmatization.lemmatize(word) for word in words]
    
    # Join the tokens back into a string
    processed_text = " ".join(words)
    
    return processed_text

@task
def preprocess_data(x_train, x_test, y_train, y_test):
    """
    Rescale the data.
    """
    vectorizer = CountVectorizer()
    x_train_vect = vectorizer.fit_transform(x_train)
    x_test_vect = vectorizer.transform(x_test)
    return x_train_vect, x_test_vect, y_train, y_test


@task
def train_model(x_train, y_train, hyperparameters):
    """
    Training the machine learning model.
    """
    clf = MultinomialNB(**hyperparameters)
    clf.fit(x_train, y_train)
    return clf


@task
def evaluate_model(model, x_train_vect, y_train, x_test_vect, y_test):
    """
    Evaluating the model.
    """
    y_train_pred = model.predict(x_train_vect)
    y_test_pred = model.predict(x_test_vect)

    train_score = metrics.accuracy_score(y_train, y_train_pred)
    test_score = metrics.accuracy_score(y_test, y_test_pred)
    
    return train_score, test_score

# Workflow

@flow(name="Naive Bayes Flow")

def workflow():
    DATA_PATH = "cleaned_data.csv"
    INPUTS = 'Review text'
    OUTPUT = 'Sentiment'
    HYPERPARAMETERS = {
                        'alpha': 1.0,
                        'fit_prior': True, 
                        'class_prior': None
                        }

    # Load data
    data = load_data(DATA_PATH)

    # Identify Inputs and Output
    x, y = split_inputs_output(data, INPUTS, OUTPUT)

    # Split data into train and test sets
    x_train, x_test, y_train, y_test = split_train_test(x, y)

    # Preprocess the data
    x_train_vect, x_test_vect, y_train, y_test = preprocess_data(x_train, x_test, y_train, y_test)

    # Build a model
    model = train_model(x_train_vect, y_train, HYPERPARAMETERS)

    # Evaluation
    train_score, test_score = evaluate_model(model, x_train_vect, y_train, x_test_vect, y_test)

    print("Train Score:", train_score)
    print("Test Score:", test_score)

# Running workflow
if __name__ == "__main__":
    
    workflow()

# Scheduling workflow
if __name__ == "__main__":
    workflow.serve(
        name="Flipkart Sentiment Prediction",
        cron="11 * * * *"
    )    