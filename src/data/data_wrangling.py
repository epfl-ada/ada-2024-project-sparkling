import json
import pandas as pd
from datetime import datetime
from src.data.load_data import *


def convert_dictionary_column_to_categorical(df_column):
    """
    Convert a colunm containing python dictionaries (in string format) to
    one hot encoding (considering every values of the directory as a category).

    Argument:
        - df_column: column of a the dataframe containing the python dictionaries

    Returns a data frame containing the one hot encoding of the different categories
    """
    set_of_unique_category = set()

    # Transform every element to dictionaries
    # Extract the values (We do not need the freebase ID)
    def get_categories(element):
        category_dictonary = json.loads(element)
        category_list = list(category_dictonary.values())
        return category_list

    # Keep the unique values from all the columns
    def apply_row_get_categories(element):
        category_list = get_categories(element)
        set_of_unique_category.update(category_list)

    df_column.apply(apply_row_get_categories)
    list_of_unique_category = list(set_of_unique_category)

    # Returning multiple columns using the apply function:
    #   https://stackoverflow.com/questions/52854147/using-result-type-with-pandas-apply-function
    def apply_row_to_one_hot(element):
        one_hot_row = pd.Series(False, index=range(len(list_of_unique_category)))
        category_list = get_categories(element)
        for category in category_list:
            category_index = list_of_unique_category.index(category)
            one_hot_row[category_index] = True

        return one_hot_row

    df_one_hot_encoding = df_column.apply(apply_row_to_one_hot)

    # Rename the columns by the name categories
    name_mapping_categories = {i: x for i, x in enumerate(list_of_unique_category)}
    df_one_hot_encoding = df_one_hot_encoding.rename(columns=name_mapping_categories)
    return df_one_hot_encoding


def clean_categories_to_one_hot(df, category_columns):
    """
    Transform all the categorical columns into one hot encoding columns

    Arguments:
        - df: Dataframe containing the categorical columns
        - category_columns: The list of columns name containing the categories

    Returns
        - (Dataframe): The same dataframe extended with the plots. Note that the given list of columns will be removed from the first extended dataset.
        - (List[DataFrame]): A list of dataframes  containing for each the associated wikipedia_ID and hot one encodings for each category.
    """
    list_dataframe_one_hot_encoding = []
    for column_name in category_columns:
        df_one_hot_encoding = convert_dictionary_column_to_categorical(df[column_name])
        df_one_hot_encoding["wikipedia_ID"] = df["wikipedia_ID"]
        list_dataframe_one_hot_encoding.append(df_one_hot_encoding)

    df = df.drop(labels=category_columns, axis=1)

    return df, list_dataframe_one_hot_encoding


def parse_release_date(df, colunm_name):
    """
    Parse every release date into three columns (release_year, release_month, release_day)

    Arguments:
        - df: Dataframe containing the colunm
        - colunm_name: The name of the colunm to parse

    Returns the same dataframe extended with the parsed date (where the original colunm is dropped)
    """

    def apply_row_parse_date(element):
        """
        For each element try to parse the date for every type of formats,
        the missing parts will be replaced by NA
        """
        result_date_array = [pd.NA] * 3
        if not pd.isna(element):
            # Specify if the format matches which part of the date is missing
            parsing_format_missing = [
                ("%Y-%m-%d", [False, False, False]),
                ("%Y-%m", [False, False, True]),
                ("%Y", [False, True, True]),
            ]
            for format, missing_one_hot in parsing_format_missing:
                try:
                    date = datetime.strptime(element, format)
                    result_date_array = [date.year, date.month, date.day]
                    # Setting the missing parts to NA
                    for i, missing in enumerate(missing_one_hot):
                        if missing:
                            result_date_array[i] = pd.NA

                    # Handling one exception
                    if result_date_array[0] == 1010:
                        result_date_array[0] = 2010

                    break
                except:
                    continue

        return pd.Series(result_date_array)

    # Splitting the release date in year, month, day columns and setting the columns to nullable int
    # https://stackoverflow.com/questions/21287624/convert-pandas-column-containing-nans-to-dtype-int
    parsed_date = df[colunm_name].apply(apply_row_parse_date).astype("Int64")
    parsed_date = parsed_date.rename(
        columns={0: "release_year", 1: "release_month", 2: "release_day"}
    )
    df = df.drop(labels=[colunm_name], axis=1)
    df = pd.concat([df, parsed_date], axis=1)

    return df


def clean_plot(df, colunm_name):
    """
    Clean the plot column

    Arguments:
        - df: Dataframe containing the colunm
        - colunm_name: The name of the colunm to clean

    Returns the same dataframe with the plot colunm cleaned
    """
    # Remove wikipedia formats

    # We replace every {{Any character}} by ''
    # (\s+|) at the end means that we also want that if there are spaces after
    # the {{...}} to count it as a match and remove it
    df[colunm_name] = df[colunm_name].str.replace("{{[^}]+}}(\s+|)", "", regex=True)

    return df


def process_df_languages(df):
    """
    Improve the list of languages in the language Dataframe

    Arguments:
        - df: The language dataframe

    Returns the same dataframe with the improved list of languages
    """
    # We want to merge couple of languages into the same language column
    # For example, there is a language "France" and also the "right" language "French Language"
    # So we are going to merge both columns, keeping only the "French Language" column.
    couple_col_replace = [
        ("French Language", "France"),
        ("Japanese Language", "Japan"),
        ("German Language", "Deutsch"),
        ("German Language", "German"),
        ("Hungarian language", "Hungary"),
        ("Italian Language", "Italian"),
    ]
    for final_col_name, remove_col_name in couple_col_replace:
        df[final_col_name] = df[final_col_name] | df[remove_col_name]
        df = df.drop(remove_col_name, axis=1)

    return df


def load_and_clean_movies_df():
    """
    Returns a list of 4 dataframes.
    - A dataframe containing the main metadata of the movies along with its plot
    - A dataframe containing the language of the movie in one hot encoding, associated to the wikipedia_ID of the movie
    - A dataframe containing the countries of the movie in one hot encoding, associated to the wikipedia_ID of the movie
    - A dataframe containing the genres of the movie in one hot encoding, associated to the wikipedia_ID of the movie
    """

    # Load data
    df_movies_metadata = load_movie_metadata()
    df_movies_plot = load_movie_plot()

    # Data wangling movie metadata
    df_movies_metadata_cleaned = parse_release_date(df_movies_metadata, "release_date")
    df_movies_metadata_cleaned, list_one_hot_encoding_dataframes = (
        clean_categories_to_one_hot(
            df_movies_metadata_cleaned, ["languages", "countries", "genres"]
        )
    )

    df_languages, df_countries, df_genres = list_one_hot_encoding_dataframes

    # Pre-process the language Dataframe
    df_languages = process_df_languages(df_languages)

    # Data wangling movie plot
    df_movies_plot_cleaned = clean_plot(df_movies_plot, "plot")

    # Merge dataframes
    df_movies = df_movies_metadata_cleaned.merge(
        df_movies_plot_cleaned, on="wikipedia_ID"
    )

    return df_movies, df_languages, df_countries, df_genres
