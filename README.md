# Disaster Response Pipeline Project

This repository contains a web app, ETL as well as ML pipeline from a dataset provided by FigureEight regarding disaster response message.

### Data Processing:
1. Clean the data and do the Extract, Transform, and Load process (ETL) so that the data can be utilized for Machine Learning pipeline. See the detail code on the `ETL_Pipeline.ipynb`
2. Create a machine learning pipeline that uses the message column to predict classifications for 36 categories (multi-output classification See the detail code on the `ML_Pipeline.ipynb`
3. Display the result on the Flask web app.


### Instructions:
1. Run the following commands in the project's root directory to set up your database and model.

    - To run ETL pipeline that cleans data and stores in database
        `python data/process_data.py data/disaster_messages.csv data/disaster_categories.csv data/DisasterMessages.db`
    - To run ML pipeline that trains classifier and saves
        `python models/train_classifier.py data/DisasterMessages.db models/classifier.pkl`

2. Run the following command in the app's directory to run your web app.
    `python run.py`

3. Go to http://0.0.0.0:3001/

### File Description:
1. The `data` folder contains the dataset, database and ETL pipeline.
2. The `models` folder contains the ML pipeline and classifier model
3. The `app` folder contains web application

### Remarks:
The trained model is not inlcuded in this repo due to the large file size.

### Acknowledgements:
- FigureEight that genereously providing the data.
- Udacity for the lessons, for providing the templates for this project, and for the answers in Knowledge (especially regarding the NaN value of the data, inconsistencies of the' 'related' column that has a value of 2, and 'child_alone' column who only has the value of 0) that helps me to clean the data. Udacity Knowledge  also provides the answer about the new HTML template and gives an idea for the additional plot.
- Various ideas and input from Stack Overflow.
