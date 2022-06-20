import sys
import pandas as pd
import numpy as np 
import sqlalchemy
import re
import nltk
nltk.download('stopwords')
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem.porter import PorterStemmer
from nltk.stem.wordnet import WordNetLemmatizer
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.pipeline import Pipeline
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import train_test_split
from sklearn.multioutput import MultiOutputClassifier
from sklearn.svm import LinearSVC
from sklearn.preprocessing import OneHotEncoder
from sklearn.metrics import accuracy_score, f1_score, average_precision_score
from sklearn import metrics
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import train_test_split
import pickle

def load_data(database_filepath):
    engine = sqlalchemy.create_engine('sqlite:///{}'.format(database_filepath))
    df = pd.read_sql_table('table_message',con=engine)
    X = df['message']
    y = df[['related', 'request', 'offer',
       'aid_related', 'medical_help', 'medical_products', 'search_and_rescue',
       'security', 'military', 'water', 'food', 'shelter', 'clothing', 'money',
       'missing_people', 'refugees', 'death', 'other_aid',
       'infrastructure_related', 'transport', 'buildings', 'electricity',
       'tools', 'hospitals', 'shops', 'aid_centers', 'other_infrastructure',
       'weather_related', 'floods', 'storm', 'fire', 'earthquake', 'cold',
       'other_weather', 'direct_report']]
    category_names = y.columns
    return X, y, category_names 

def tokenize(text):
    #cleaning
    text= text.replace(r'://www.([\w\-\.]+\S+)','') #replace URL
    text= text.replace(r'[^\w\s]|\b\w{1,2}\b|\d+','') #remove digit, less than 2 chars

    tokens = word_tokenize(text)
    lemmatizer = WordNetLemmatizer()
    
    st = PorterStemmer()
    lemmatizer = WordNetLemmatizer()
    stop= stopwords.words('english')
    #stop = words #I will not remove for now the stop words since they convey some meaning 
    stop = [st.stem(x) for x in stop]

    clean_tokens = []
    for tok in tokens:
        clean_tok = lemmatizer.lemmatize(tok).lower().strip() 
        clean_tokens.append(clean_tok)

    return clean_tokens    
 

def build_model():
    #Pipeline 
    pipeline = Pipeline([
        ('vect', CountVectorizer(tokenizer=tokenize)),
        ('tfidf', TfidfTransformer()),
        ('classifier',MultiOutputClassifier(LinearSVC()))
    ])     

    param_grid = {
    # try different feature engineering parameters
    'classifier__estimator__penalty': ['l2','l1'],
    'classifier__estimator__C': [1,10,100],
    }

    grid_search = GridSearchCV(pipeline, param_grid,
                           cv=5, n_jobs=-1, scoring = 'f1')
    return grid_search


def evaluate_model(model, X_test, Y_test, category_names):
    y_pred = model.predict(X_test)
    print (metrics.classification_report(Y_test,y_pred,target_names = category_names))


def save_model(model, model_filepath):
    filename = '{}'.format(model_filepath)
    pickle.dump(model, open(filename, 'wb'))


def main():
    if len(sys.argv) == 3:
        database_filepath, model_filepath = sys.argv[1:]
        print('Loading data...\n    DATABASE: {}'.format(database_filepath))
        X, y, category_names = load_data(database_filepath)
        X_train, X_test, Y_train, Y_test = train_test_split(X, y, test_size=0.2)
        
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