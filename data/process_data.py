import sys
import pandas as pd
import numpy as np
from sqlalchemy import create_engine


def load_data(messages_filepath, categories_filepath):
    """
    Load messages and categories datasets and merge them into a single dataframe.

    Parameters:
    messages_filepath (str): Filepath to the messages CSV file.
    categories_filepath (str): Filepath to the categories CSV file.

    Returns:
    df (pd.DataFrame): Merged dataframe containing messages and categories.
    """
    # Load messages dataset
    messages = pd.read_csv(messages_filepath)

    # Load categories dataset
    categories = pd.read_csv(categories_filepath)

    # Merge datasets on the 'id' column
    df = messages.merge(categories, on='id')

    return df


def clean_data(df):
    """
    Clean the merged dataframe by splitting categories, converting values to binary, and removing duplicates.

    Parameters:
    df (pd.DataFrame): Merged dataframe containing messages and categories.

    Returns:
    df (pd.DataFrame): Cleaned dataframe ready for saving to the database.
    """
    # Split the categories column into separate columns
    categories = df['categories'].str.split(';', expand=True)

    # Use the first row to extract category names
    row = categories.iloc[0]
    category_colnames = row.apply(lambda x: x.split('-')[0])

    # Rename columns with extracted category names
    categories.columns = category_colnames

    # Convert category values to binary (0 or 1)
    for column in categories:
        # Set each value to be the last character of the string
        categories[column] = categories[column].astype(str).str[-1]

        # Convert column from string to numeric
        categories[column] = categories[column].astype(int)

    # Drop the original 'categories' column from df
    df = df.drop(columns=['categories'])

    # Concatenate df with the new categories dataframe
    df = pd.concat([df, categories], axis=1)

    # Remove duplicates
    df = df.drop_duplicates()

    # Remove duplicate columns from the DataFrame
    df = df.loc[:, ~df.columns.duplicated()]

    return df


def save_data(df, database_filename):
    """
    Save the cleaned dataframe to an SQLite database.

    Parameters:
    df (pd.DataFrame): Cleaned dataframe to be saved.
    database_filename (str): Filepath to the SQLite database.
    """
    # Create the database engine
    engine = create_engine(f'sqlite:///{database_filename}')

    # Save the dataframe to an SQL database
    df.to_sql('disaster_response', engine, index=False, if_exists='replace')


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
        print('Please provide the filepaths of the messages and categories '
              'datasets as the first and second argument respectively, as '
              'well as the filepath of the database to save the cleaned data '
              'to as the third argument. \n\nExample: python process_data.py '
              'disaster_messages.csv disaster_categories.csv '
              'DisasterResponse.db')


if __name__ == '__main__':
    main()
