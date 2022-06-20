# Disaster_messages_classification
Project under Udacity's Data Science Nanodegree - Data Engineering part.
Data is provided from the program. The purpose of the project is to set up a machine learning pipeline for multi-label classification of disaster messages into different categories, deploy that pipeline on a web service and create a visualisation dashboard. 

## Installations 
For reference - check the requirements.txt file in the main folder. 

## Project Motivation
Project under Udacity's Data Science Nanodegree - Data Engineering part.
The purpose of the project is to demonstrate abilities for setting up an ETL pipeline, a ML pipeline, and set up a dashboard to communicate the results. 

## File Descriptions
The project contains the following folders and files: 

- workspace #this folder contains all the files for the web visualisation dashboard
|- app
| |- template
| | |- master.html  # main page of web app
| | |- go.html  # classification result page of web app
| |- run.py  # Flask file that runs app

|- data
| |- disaster_categories.csv  # data to process 
| |- disaster_messages.csv  # data to process
| |- process_data.py        # file to clean and transform the initial data 
| |- ResponseDisaster.db   # database to save clean data to

| - models
| |- train_classifier.py # file to build the machine learning pipeline
| |- classifier.pkl  # saved model 

- notebooks 
| - 01.01.-ag-initial-analysis-cleaning.ipynb # data cleaning, transformation and EDA of the initial data
| - 02.01.-ag-data-cleaning.ipynb # the notebook deals with the text processing and text preparation to be transformed into numerical data
  - 02.02.-ag-data-cleaning-text-preprocessing-spacy.ipynb  # it is a complementary notebook that used spacy library and its nlp pipeline for all text processing tasks; 
  - 03.01.-ag-model-pipeline.ipynb # this notebook contains the ML pipeline
  
- data
| - raw data # the raw files
| | - disaster_messages.csv
| | - disaster_categories.csv
| - processed data # the files processed throughtout the different steps of the project
| | - data_after_eda.pkl # the file with the cleaned data, output from 01.01. notebook
| | - data_after_prcessing_spacy.pkl # output from notebook 02.02.; suplementary file
| | - data_after_text_processing.pkl # output from notebook 02.01.; file with text preprossed data
| - preprocess_help_functions.py #file that contains some additional functions and transformations tested on the data - e.g. translation, language detection etc.

- README.md

## How to Interact with your project/Results
TBased on the analysis performed, we have seen some interesting information about the AirBNBs in Boston:

- beginning of Autumn is the busiest period for the airBNBs occupancy rates
- there are neighbourhoods that reveal a capacity for new AirBNBs.
- a more precise modelling and data mining is necessary to build more robust model, that explains better what affect the prices and its values.

## Licensing, Authors, Acknowledgements, etc.
I have used some of the code from the lectures from the python files.