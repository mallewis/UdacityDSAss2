# import libraries
import pandas as pd
import sqlalchemy
import sys


def load_data(messages_filepath, categories_filepath):
    
   '''
   Input: Messages & categories csv files
   
   Process:
   -Read both csvs as pandas dataframes
   -Merge/join both dfs on ID column
   
   Output: Merged dataframes
   '''
    # read csvs as pandas dfs
    messages = pd.read_csv(messages_filepath)
    categories =  pd.read_csv(categories_filepath)
    
    # merge datasets (join) on ID column.
    df = messages.merge(categories, on='id')
    return df


def clean_data(df):
    '''
    input: merged dataframe
    
    process:
    -split column into 36 individual columns based on ;
    -Extract column names & clean
    -clean each value to numeric only
    -Apply new column names to cleaned data
    -remove duplicates
    
    Output: cleaned, splitout messages classified dataframe
    '''
    
    # create a dataframe of the 36 individual category columns
    #split into columns on the semi-colon ';'   
    categories = df.categories.str.split(pat=';', expand=True)
    # select the first row of the categories dataframe
    row = categories.iloc[0]
    
    # use this row to extract a list of new column names for categories.
    # one way is to apply a lambda function that takes everything 
    # up to the second to last character of each string with slicing
    category_colnames = row.map(lambda x: x[:-2])

    # rename the columns of `categories`
    categories.columns = category_colnames
    
    for column in categories:
        # set each value to be the last character of the string
        categories[column] = categories[column].astype(str).map(lambda x: x[-1])

        # convert column from string to numeric
        categories[column] = categories[column].astype(int)
        
    # drop the original categories column from `df`
    del df['categories']
    
    # concatenate the original dataframe with the new `categories` dataframe
    df = pd.concat([df, categories],  join='inner', axis=1)
    
    # drop duplicates
    df.drop_duplicates(inplace=True)
    
    return df

def save_data(df, database_filename):
    #Create database and import pandas dataframe to db as table called messages
    engine = sqlalchemy.create_engine('sqlite:///' + database_filename) #'sqlite:///pipeline.db')
    df.to_sql('messages', engine, index=False)


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