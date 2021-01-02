# import libraries
import sys
import pandas as pd
import re
import numpy as np
import sqlalchemy
import pickle

#NLTK libraries
import nltk
nltk.download('punkt')
nltk.download('stopwords')
nltk.download('wordnet')
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer
from nltk.stem.wordnet import WordNetLemmatizer

#SK Learn Libraries
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer
from sklearn.ensemble import RandomForestClassifier,  AdaBoostClassifier
from sklearn.pipeline import Pipeline,  FeatureUnion
from sklearn.model_selection import train_test_split
from sklearn.multioutput import MultiOutputClassifier
from sklearn.metrics import confusion_matrix
from sklearn.metrics import classification_report
from sklearn.model_selection import GridSearchCV
from sklearn.feature_extraction.text import HashingVectorizer


def load_data(database_filepath):
    '''
    Input: Database containing cleaned message data
    Process:
    - Reads sqllite database messages table.
    - Splits the data into 32 catagories and the already classified information. (x, y)
    Output: x & y datasets and lists of 32 categories names
    '''
    engine = sqlalchemy.create_engine('sqlite:///' + database_filepath)
    df = pd.read_sql_table('messages', engine)  # read SQL table into pandas dataframe
    x = df.message.values  # Messge column only
    y = df.drop(['id', 'message', 'original', 'genre'], axis = 1) #drop the following columns from df: 'id', 'message', 'original', 'genre'
    category_names = list(y.columns.values) #get a lists of category names from columns
    return x, y, category_names


def tokenize(text):
    '''
    Input: Message text
    
    Process:
    -puts messages into list.
    -remove non-alphanumeric characters
    -make all lower case
    -remove stopwords
    -lemmetize & stem words
    
    Output: Returns tokenized text
    '''
    text = text.lower() # make text lower case
    text = re.sub(r"[^a-zA-Z0-9]", " ", text) # remove everything but text and numbers
    text = word_tokenize(text) # tokenize words to break them into individual words
    text = [t for t in text if t not in stopwords.words("english")] # remove stops words sycg as 'a, an, in'
    text = [PorterStemmer().stem(t) for t in text] # stem and lemmetize words to bring them to the root form of the word
    text = [WordNetLemmatizer().lemmatize(t, pos='v') for t in text]
    
    return text


def build_model():
    '''
    Input: None
    
    Process:
    -build pipeline of transformers and classifier, with multi-output classes.
    -Apply grid search to find optimum parameters
    
    Output: pipeline
    '''
    # pipeline
    pipeline = Pipeline([
        ('vect', CountVectorizer(tokenizer=tokenize)), # Creates a matrix that counts the words
        ('tfidf', TfidfTransformer()), # divides the above count per message.
        ('clf',  MultiOutputClassifier(RandomForestClassifier())) # runs a classifier that uses decision tree to attempt classify the messages
    ])

    # cv grid search parameters 
    parameters ={
    'tfidf__norm' : ['l1', 'l2']
    }
    
    cv = GridSearchCV(pipeline, param_grid=parameters) #Run grid search on parameters
    
    return cv

def evaluate_model(model, X_test, Y_test, category_names):
    '''
    Input: ML model, x test, y test and 32 category names
    
    Process:
    -Predict the x test data using the model
    -loop/enumerate each category and print out classification report to indicate accuracy.
    
    Output: Print of classifier
    '''
    #predict the X test data
    y_pred = model.predict(X_test)
   # loop & enumerate through each item in 'y' so that is runs the classification report on each class - testing the accuracy
    for i, col in enumerate(category_names):
        print("------" + col)
        col_y_test = list(Y_test.values[:,i])
        col_y_pred = list(y_pred[:, i])
        classRep = classification_report(col_y_test, col_y_pred)
        print(classRep)
        
def save_model(model, model_filepath):
    #exports ML pipline/model to PKL file.
    pickle.dump(model.best_estimator_, open(model_filepath, 'wb'))


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