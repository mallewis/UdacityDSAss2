import json
import plotly
import pandas as pd

from nltk.stem import WordNetLemmatizer
from nltk.tokenize import word_tokenize

from flask import Flask
from flask import render_template, request, jsonify
from plotly.graph_objs import Bar
from plotly.graph_objs import Pie
from sklearn.externals import joblib
from sqlalchemy import create_engine


app = Flask(__name__)

def tokenize(text):
    tokens = word_tokenize(text)
    lemmatizer = WordNetLemmatizer()

    clean_tokens = []
    for tok in tokens:
        clean_tok = lemmatizer.lemmatize(tok).lower().strip()
        clean_tokens.append(clean_tok)

    return clean_tokens

# load data
engine = create_engine('sqlite:///../data/DisasterResponse.db')
df = pd.read_sql_table('messages', engine)

# load model
model = joblib.load("../models/classifier.pkl")


# index webpage displays cool visuals and receives user input text for model
@app.route('/')
@app.route('/index')
def index():
    
    # extract data needed for visuals
    
    #Data for bar chart No1 based on genres.
    genre_counts = df.groupby('genre').count()['message'] #counts for genres
    genre_names = list(genre_counts.index) #genre titles
    
   
   #Bar chart No2: 36 message classifications counts
    
    #Sum the values for each message classifications
    type_counts = df[['related', 'request', 'offer', 'aid_related', 'medical_help', 
                   'medical_products', 'search_and_rescue', 'security', 'military', 
                   'child_alone', 'water', 'food', 'shelter', 'clothing', 'money', 
                   'missing_people', 'refugees', 'death', 'other_aid', 
                   'infrastructure_related', 'transport', 'buildings', 'electricity', 
                   'tools', 'hospitals', 'shops', 'aid_centers', 'other_infrastructure', 
                   'weather_related', 'floods', 'storm', 'fire', 'earthquake', 'cold', 
                   'other_weather', 'direct_report']].sum()
    
    # Get a lists of 36 classification for messages
    type_names = list(type_counts.index)
    #remove underscores to tidy up for plot labelling
    type_names = [tn.replace('_', ' ') for tn in type_names] 
    
    #Pie chart: show related messages vs non related messages
    
    #clean values and summaries related vs non-related
    rel_counts = df.loc[df['related'] != 2]
    rel_counts = rel_counts['related'].value_counts()
    #create label names on pie chart
    rel_names = ['Yes', 'No']
    
    # create visuals
    graphs = [
        {
            'data': [
                Bar(#Bar chart No1
                    x=genre_names,
                    y=genre_counts
                )
            ],

            'layout': {
                'title': 'Distribution of Message Genres',
                'yaxis': {
                    'title': "Count"
                },
                'xaxis': {
                    'title': "Genre"
                }
            }
        },
        {
            'data': [
                Bar(#Bar chart No2
                    x=type_names,
                    y=type_counts
                )
            ],

            'layout': {
                'title': 'Count of Message Per Class',
                'yaxis': {
                    'title': "Count"
                },
                'xaxis': {
                    'title': "Message Classification",
                    'tickangle': 45
                }
            }
        },
        {
            'data': [
                Pie(#Pie chart
                    labels=rel_names,
                    values=rel_counts
                )
            ],

            'layout': {
                'title': 'Proportion of Related Messages'
                
            }
        }
    ]
    
    # encode plotly graphs in JSON
    ids = ["graph-{}".format(i) for i, _ in enumerate(graphs)]
    graphJSON = json.dumps(graphs, cls=plotly.utils.PlotlyJSONEncoder)
    
    # render web page with plotly graphs
    return render_template('master.html', ids=ids, graphJSON=graphJSON)


# web page that handles user query and displays model results
@app.route('/go')
def go():
    # save user input in query
    query = request.args.get('query', '') 

    # use model to predict classification for query
    classification_labels = model.predict([query])[0]
    
    classification_results = dict(zip(df.columns[4:], classification_labels))
    print("aaaa", classification_results)

    # This will render the go.html Please see that file. 
    return render_template(
        'go.html',
        query=query,
        classification_result=classification_results
    )


def main():
    app.run(host='0.0.0.0', port=3001, debug=True)


if __name__ == '__main__':
    main()