import sys
import re
import pandas as pd
import numpy as np
import sqlite3
from sqlalchemy import create_engine

import pickle
import joblib


# download necessary NLTK data
import nltk
nltk.download(['punkt', 'wordnet', 'stopwords', 'averaged_perceptron_tagger'])
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer
from nltk.corpus import stopwords

from sklearn.pipeline import Pipeline
from sklearn.metrics import confusion_matrix, classification_report
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline, FeatureUnion
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.ensemble import RandomForestClassifier
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.multioutput import MultiOutputClassifier
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer


def load_data(database_filepath):
    '''
    Function to load dataframe from SQL database
    Input: database_filepath(str) - path to SQL database
    Output:
    X = message dataset
    Y = labels of categories dataset
'   '''

    #engine = create_engine('sqlite:///DisasterResponse.db')
    #df = pd.read_sql("SELECT * FROM messages", engine)
    #file_path = '../data/DisasterResponse.db'
    conn = sqlite3.connect(database_filepath)
    df = pd.read_sql('SELECT * FROM messages', conn)

    # create X and  Y dataframe
    X = df['message']
    Y = df[df.columns[3:]]
    category_names = Y.columns.values

    return X,Y, category_names

def tokenize(text):
    '''
    Function to tokenize, lemmatize, and normalize text
    Input: text (str) - text to be processed
    Ouput: cleant_token(str) - list of clean tokens
    '''
    url_regex = 'http[s]?://(?:[a-zA-Z]|[0-9]|[$-_@.&+]|[!*\(\),]|(?:%[0-9a-fA-F][0-9a-fA-F]))+'

    #remove stop words
    stop_words = stopwords.words("english")

    # get list of all urls using regex
    detected_urls = re.findall(url_regex, text)

    # replace each url in text string with placeholder
    for url in detected_urls:
        text = text.replace(url, 'urlplaceholder')

    # extract word tokens from the text - tokenize text
    tokens = word_tokenize(text)

    # initiate lemmatizer
    lemmatizer = WordNetLemmatizer()

    # iterate through each token
    clean_tokens = []
    for tok in tokens:
        if tok not in stop_words:
            # lemmatize, normalize case, and remove leading/trailing white space
            clean_tok = lemmatizer.lemmatize(tok).lower().strip()
            clean_tokens.append(clean_tok)

    return clean_tokens

# Create a custom transformer to extract verb
class StartingVerbExtractor(BaseEstimator, TransformerMixin):
    '''
    A class verb extractor - an estimator to extract the starting verb in a sentence
    '''
    def starting_verb(self, text):
        """
        Function to evaluate whether there is a verb starting a sentence
        in the text.
        INPUT:
        text (str) - text to search the verb
        """
        # tokenize by sentences
        sentence_list = nltk.sent_tokenize(text)

        for sentence in sentence_list:
            # tokenize each sentence into words and tag part of speech
            pos_tags = nltk.pos_tag(tokenize(sentence))
            if len(pos_tags) > 0:
            # index pos_tags to get the first word and part of speech tag
                first_word, first_tag = pos_tags[0]

        # return true if the first word is an appropriate verb or RT for retweet
                if first_tag in ['VB', 'VBP'] or first_word == 'RT':
                    return True
        return False

    def fit(self, x, y=None):
        """
        Fit function is required for an estimator.
        """
        return self

    def transform(self, X):
        """
        Transfrom function to transform the data
        Running self.starting_verb functin and return the data as X
        """
        # apply starting_verb function to all values in X
        X_tagged = pd.Series(X).apply(self.starting_verb)
        return pd.DataFrame(X_tagged)


def build_model():
    '''
    Function to return model that has been pipelined
    '''
    #build pipeline
    pipeline = Pipeline([
        ('features', FeatureUnion([
            ('text_pipeline', Pipeline([
                ('vect', CountVectorizer(tokenizer=tokenize)),
                ('tfidf', TfidfTransformer())
            ])),

            ('starting_verb', StartingVerbExtractor())
        ])),

        ('clf', MultiOutputClassifier(RandomForestClassifier()))
    ])

    #specifity the parameter for grid search
    parameters = {
    #'features__text_pipeline__tfidf__use_idf':(True, False),
    #'clf__estimator__n_estimators': [10,50],
    'clf__estimator__min_samples_split': [2, 4]
        }

    # create grid search object
    model = GridSearchCV(pipeline, parameters, n_jobs=1, verbose=1)

    return model


def evaluate_model(model, X_test, Y_test, category_names):
    '''
    Function to evaluate the built model and print out classification report
    Input:
    - model (model object)
    - X_test - array containting message dataset
    - Y_test - array containing category dataset
    -category_names - a list of category names
    Output: classification report
    '''
    Y_pred_test = model.predict(X_test)
    #Y_pred_train = model.predict(X_train)

    #classification report on the test data
    print(classification_report(Y_test.values, Y_pred_test, target_names=category_names))
    print('--------------------------------------------------------------------')
    #classification report on the train data
    #print(classification_report(Y_train.values, Y_pred_train, target_names=category_names))

def save_model(model, model_filepath):
    '''
    Function to save model to pickle file
    input:
    model (model object) - the model to be saved
    model_filepath (str) - saving path for the model
    '''

    file = open(model_filepath, 'wb')
    pickle.dump(model, file)
    file.close()

def main():
    '''
    Funciton to run the main function
    '''
    if len(sys.argv) == 3:
        database_filepath, model_filepath = sys.argv[1:]
        print('Loading data...\n    DATABASE: {}'.format(database_filepath))
        X, Y, category_names = load_data(database_filepath)
        X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2)
        
        print('Building model...')
        model = build_model()
        
        print('Training model...')
        model.fit(X_train, Y_train)
        
        print('Evaluating model...')
        evaluate_model(model, X_test, Y_test, category_names)

        print('Saving model...\n    MODEL: {}'.format(model_filepath))
        save_model(model, model_filepath)

        print('Trained model saved!')

    else:
        print('Please provide the filepath of the disaster messages database '\
              'as the first argument and the filepath of the pickle file to '\
              'save the model to as the second argument. \n\nExample: python '\
              'train_classifier.py ../data/DisasterResponse.db classifier.pkl')


if __name__ == '__main__':
    main()