import sys
import pandas as pd
import numpy as np
from sqlalchemy import create_engine


def load_data(messages_filepath, categories_filepath):
    '''
    Function to load and read 2 data file: disaster message csv and categories csv
    returning a merged dataframe

    Input:
    message_filepath(str) - the filepath to message csv
    categories_filepath(str) - the filepath to categories csv

    Output: merged dataframe of message and categories
    '''
    messages = pd.read_csv(messages_filepath)
    categories = pd.read_csv(categories_filepath)
    df = messages.merge(categories, on = 'id')

    return df

def clean_data(df):
    '''
    Function to transform categories, clean data,
    drop duplicates, drop unnecessary column (with NaN value),
    and inconsistent value

    Input: initial df (Pandas DataFrame)

    Ouput: clean df (Pandas DataFrame)
    '''
    #  1 split `categories` into separate category columns.
    # create a dataframe of the 36 individual category columns
    categories = df['categories'].str.split(';', expand = True)

    # 2 extract a list of new column names for categories by applying a lambda function
    row = categories.iloc[0]
    category_colnames = row.str.split('-').apply(lambda x:x[0]).to_list()
    categories.columns = category_colnames

     # 3 convert category values to  numbers 0 or 1
    for column in categories:
        # set each value to be the last character of the string
        categories[column] = categories[column].transform(lambda x: x[-1:])

        # convert column from string to numeric
        categories[column] = pd.to_numeric(categories[column])
    #categories = categories.applymap(lambda value: int(value[-1]))
    # 4 modify value of 'related' column to 0 and 1 only, since it has value 2
    categories['related'] = categories['related'].apply(lambda value: value%2)

    # 5 Replace `categories` column in `df` with new category columns.
    # dropping categories and original columns (the latter has a lot of nan values)
    df_clean  = df.drop(['categories', 'original'] ,axis = 1)

    # concatenate the original dataframe with the new `categories` dataframe
    df_clean = pd.concat([df_clean, categories], axis = 1)

    # 6 Remove duplicates
    df_clean  = df_clean.drop_duplicates()

    return df_clean

def save_data(df, database_filename):
    '''
    Function to save clean dataset into sqlite database
    Input: df (Pandas DataFrame)
    Output: database_filename (str)
    '''
    engine = create_engine('sqlite:///{}'.format(database_filename))
    df.to_sql('messages', engine, index=False, if_exists='replace')


def main():
    '''
    Function to run the main function
    '''
    if len(sys.argv) == 4:

        messages_filepath, categories_filepath, database_filepath = sys.argv[1:]

        print('Loading data...\n    MESSAGES: {}\n    CATEGORIES: {}'
              .format(messages_filepath, categories_filepath))
        df = load_data(messages_filepath, categories_filepath)

        print('Cleaning data...')
        df = clean_data(df)
        
        print('Saving data...\n    DATABASE: {}'.format(database_filepath))
        save_data(df, database_filepath)
        
        print('Cleaned data saved to database!')
    
    else:
        print('Please provide the filepaths of the messages and categories '\
              'datasets as the first and second argument respectively, as '\
              'well as the filepath of the database to save the cleaned data '\
              'to as the third argument. \n\nExample: python process_data.py '\
              'disaster_messages.csv disaster_categories.csv '\
              'DisasterResponse.db')


if __name__ == '__main__':
    main()