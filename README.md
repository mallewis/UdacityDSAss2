### Title:  Udacity - Data Science - Assignment 2 - Disaster Response Pipeline Project

### Introduction:
This aim of this project is to demonstrate data pipelines, from extract, transform, load (ETL), 
machine learning and presenting analysis results in a web application.

### Project Motivation:
This project is important to me as it allows me to understand the data pipeline process:
ETL:
1. Data Cleaning
2. Data Wrangling
3. Exploratory analysis
4. Load to formal database

Machine Learning:
5. Split data
6. build ML pipeline
7. Train data
8. Refine model
9. Export model

Web App:
10. Understanding web applications (front/backend)
11. Data visualizations

### Technology:
- Python (pandas, numpy, sqlalchemy, sklearn).
- Python Flask framework.
- HTML, CSS, Javascript
- Jupyter Notebooks
- Git/github
- sqllite

### Repository contents:

/app:
- templates (folder: contain go & master .html files)
- run.py - python file that runs the web app using flask framework.
- HomePage_graphs.png - Screenshot of web app home page with additional data visualizations
- ML_Model_Results.png - Screenshot of web app ML analysis results page showing message classifications

/data:
- DisasterResponse.db - cleaned data stored in sqllite database
- disaster_categories.csv - raw csv file for disaster categories
- Disaster_messages.csv - raw csv file for disaster messages
- process_data.py - Python file to clean, tranform and load data (ETL)
- ETL Pipeline Preparation.ipynb - Working ETL Jupyter note book using to create process_data.py

/models:
- classifier.pkl - pkl file that stores the machine learning code for the web app.
- train_classifier.py - python file to load, build and train ML pipeline.
- ML_Pipeline.ipynb - Working ML Pipeline Jupyter notebook, used to make train_classifier.py


### Instructions:
1. Run the following commands in the project's root directory to set up your database and model.

    - To run ETL pipeline that cleans data and stores in database
        `python data/process_data.py data/disaster_messages.csv data/disaster_categories.csv data/DisasterResponse.db`
    - To run ML pipeline that trains classifier and saves
        `python models/train_classifier.py data/DisasterResponse.db models/classifier.pkl`

2. Run the following command in the app's directory to run your web app.
    `python run.py`

3. Go to http://0.0.0.0:3001/

4. On the web app home page, you will see 3 graphs displaying statistics about the message classification training dataset.

5. To run the ML analyis against a new message, type in the message in the dialog box and click 'classify message'.
   This will take you to a new page that highlights the categories in green that the ML model has classified the data.
   Use message example:  "Please, we need tents and water. We are in Silo, Thank you!" to see tools capabilities.

### Acknowledgements & References:
 - The dataset is from Twitter messages.
 - This has been extracted and made available from Figure Eight & Udacity


The following resources were used to formulate the code and analysis:
 - https://stackoverflow.com/
 - https://www.kite.com/python
 - https://pandas.pydata.org/
 - https://plotly.com/python/
 - https://github.com/
 - https://www.codegrepper.com/
 - https://scikit-learn.org/stable/
 - https://www.nltk.org/
 - https://classroom.udacity.com/
