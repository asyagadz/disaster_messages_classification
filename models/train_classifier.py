import sys


def load_data(database_filepath):
    engine = sqlalchemy.create_engine('sqlite:///{}'.format(database_filepath))
    df = pd.read_sql_table('table_message',con=engine)


def tokenize(serie):
    '''Function to normalize data, clean special characters,clean rows with only empty strings, noise
    '''
    #lower case
    serie = serie.astype(str).str.lower()
    #cleaning
    serie= serie.str.replace(r'://www.([\w\-\.]+\S+)','') #replace URL
    serie= serie.str.replace(r'[^\w\s]|\b\w{1,2}\b|\d+','') #remove digit, less than 2 chars
     #define stemmer
    st = PorterStemmer()
    lemmatizer = WordNetLemmatizer()
    stop= stopwords.words('english') + words
    #stop = words #I will not remove for now the stop words since they convey some meaning 
    stop = [st.stem(x) for x in stop]
    #define stemmer
    st = PorterStemmer()
    #apply tokenization & lemmatization
    serie= serie.apply(lambda x: " ".join([lemmatizer.lemmatize(word) for word in x.split() 
                                           if st.stem(word) not in stop])) 
    return serie

def build_model():
    df['tokens'] = tokenize(df['message'])
    X = df[['tokens','genre']].copy()
    y = df[['related', 'request', 'offer',
       'aid_related', 'medical_help', 'medical_products', 'search_and_rescue',
       'security', 'military', 'water', 'food', 'shelter', 'clothing', 'money',
       'missing_people', 'refugees', 'death', 'other_aid',
       'infrastructure_related', 'transport', 'buildings', 'electricity',
       'tools', 'hospitals', 'shops', 'aid_centers', 'other_infrastructure',
       'weather_related', 'floods', 'storm', 'fire', 'earthquake', 'cold',
       'other_weather', 'direct_report']].copy()

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.10,random_state=42)   

    #Pipeline 
    preprocessor = ColumnTransformer(
        [('genre_cat', OneHotEncoder(dtype='int', handle_unknown='ignore'), ['genre']),
        ('description_tfidf', TfidfVectorizer( ngram_range=(1,1)))],
        remainder='passthrough')

    pipeline = Pipeline([
        ('preprocessor',preprocessor),
        ('classifier',MultiOutputClassifier(LinearSVC()))
    ])     

    pipeline.fit(X_train, y_train)


def evaluate_model(model, X_test, Y_test, category_names):
    y_pred = model.predict(X_test)
    metrica = metrics.classification_report(Y_test,y_pred,target_names = y_test.columns.values)
    metrica = pd.DataFrame(metrica).transpose()
    for category in category_names:
        print (metrica[metrica.index==category])


def save_model(model, model_filepath):
    filename = '{}.pkl'.format(model_filepath)
    pickle.dump(model, open(filename, 'wb'))


def main():
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