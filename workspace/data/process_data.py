import sys
import pandas as pd
import numpy as np
import sqlalchemy


def load_data(messages_filepath, categories_filepath):
    messages = pd.read_csv(messages_filepath)
    categories = pd.read_csv(categories_filepath)
    df = messages.merge(categories, how = 'inner', on = 'id')
    return df


def clean_data(df):
    #split the values in different columns
    categories_new = df['categories'].str.split(pat=';',expand=True)
    #change the column names 
    categories_new.columns = categories_new.iloc[0]
    #remove numbers from column names
    new_list=[]
    for col in categories_new.columns:
        new_col = col.split('-')[0]
        new_list.append(new_col)
    categories_new.columns = new_list
    #trannsform the values into binary
    for col in categories_new.columns:
        categories_new[col]=categories_new[col].apply(lambda x:x.split('-')[1])
        categories_new[col]=pd.to_numeric(categories_new[col])
    #merge back to the original data
    df = df.join(categories_new)
    #drop the original column 
    df.drop(['categories'],axis=1,inplace=True)
    #drop duplicates
    df.drop_duplicates(subset='id',inplace=True)
    #drop other columns
    df.drop(columns=['original'],axis=1,inplace=True)
    #change labels in related 
    df.loc[df['related']==2,'related']=1
    #drop columns with only 1 value - constant categories
    col_list = []
    for col in df.columns:
        if df[col].nunique()==1:
            col_list.append(col)
    df.drop(col_list,axis=1,inplace=True)
    return df

def save_data(df,database_filepath):
    engine = sqlalchemy.create_engine('sqlite:///{}'.format(database_filepath))
    df.to_sql('table_message', engine,if_exists='replace')


def main():
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