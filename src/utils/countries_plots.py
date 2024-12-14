import pandas as pd

COLORS = {
    'joy': '#f3c35f',
    'anger': '#e13c29',
    'disgust': '#8bc664',
    'sadness': '#85b8ed',
    'fear': '#b99be5',
    'surprise': '#de9f81'
}


def map_year_to_decenny(df):
    """
    Adds a column map_year which will map the years to their decenny
    """
    
    df_with_decenny = df.copy(deep=True)
    df_with_decenny['decenny'] = (df_with_decenny['release_year'] // 10) * 10
    
    return df_with_decenny


def merge_emotions_and_movies(df1, df2):
    """
    Parameters 
        df1 : dataframe of movies 
        df2 : dataframe of countries of movies

    Adds row(s) for each movie with one of its countries
    """
    
    df2_long = df2.melt(id_vars=['wikipedia_ID'], 
                    var_name='Country', 
                    value_name='is_in_country')

    df2_long = df2_long[df2_long['is_in_country'] == True].drop(columns='is_in_country')

    df_with_countries = pd.merge(df1, df2_long, on='wikipedia_ID', how='inner')

    return df_with_countries


def get_most_dominant_emotion(df, reviews=False):
    """
    Parameters:
        df: Dataframe containing emotion scores.
        reviews (bool): If True, uses review-based emotion columns; otherwise, uses plot-based columns.

    Returns:
        Dataframe: A copy of the input Dataframe with an added 'dominant_emotion' column.
    """
    df = df.copy(deep=True)

    if reviews:
        emotion_columns = [
            'review_anger',
            'review_disgust',
            'review_fear',
            'review_joy',
            'review_sadness',
            'review_surprise'
        ]
        prefix_to_remove = 'review_'
    else:
        emotion_columns = [
            'normalized_plot_anger_without_neutral',
            'normalized_plot_disgust_without_neutral',
            'normalized_plot_fear_without_neutral',
            'normalized_plot_joy_without_neutral',
            'normalized_plot_sadness_without_neutral',
            'normalized_plot_surprise_without_neutral'
        ]
        prefix_to_remove = 'normalized_plot_'
        suffix_to_remove = '_without_neutral'

    #getting the column with the maximum value
    df['dominant_emotion'] = (
        df[emotion_columns]
        .idxmax(axis=1)  
        .str.replace(prefix_to_remove, '', regex=False)
        .str.replace(suffix_to_remove, '', regex=False) if not reviews else
        df[emotion_columns]
        .idxmax(axis=1)
        .str.replace(prefix_to_remove, '', regex=False)
    )
    
    return df




def get_accurate_countries(row):
    """
    Replaces each country that does not exist anymore in 2024 by the current country
    """
    if row['Country'] in ['Palestinian territories', 'Palestinian Territories', 'Mandatory Palestine']:
        return 'Palestine'
    elif row['Country'] == 'Kingdom of Italy':
        return 'Italy'
    elif row['Country'] == 'Georgian SSR':
        return 'Georgia'
    elif row['Country'] in ['German Democratic Republic', 'West Germany', 'Weimar Republic']:
        return 'Germany'
    elif row['Country'] == 'Soviet Union':
        return 'Russia'
    elif row['Country'] == 'Kingdom of Great Britain':
        return 'United Kingdom'
    return row['Country']


def get_decenny_and_country(df_movies_with_emotions_normalized, df_countries):
    """
    Returns the df_movies but with the country and decenny
    """
    df_with_decenny = map_year_to_decenny(df_movies_with_emotions_normalized)
    df_with_countries = merge_emotions_and_movies(df_with_decenny, df_countries)
    df_with_countries['Country'] = df_with_countries.apply(get_accurate_countries, axis=1)
    return df_with_countries



def average_emotions_per_country(df, reviews=False):
    """
    Returns the mean emotion score based on the couple country/decenny for a dataframe
    """
    if reviews:
        emotion_columns = [
            'review_anger',
            'review_disgust',
            'review_fear',
            'review_joy',
            'review_sadness',
            'review_surprise'
        ]
    else:
        emotion_columns = [
            "normalized_plot_anger_without_neutral",
            "normalized_plot_disgust_without_neutral",
            "normalized_plot_fear_without_neutral",
            "normalized_plot_joy_without_neutral",
            "normalized_plot_sadness_without_neutral",
            "normalized_plot_surprise_without_neutral",
        ]
    
    grouped_df = (
        df.groupby(["Country", "decenny"])[emotion_columns]
        .mean()
        .reset_index()
    )
    
    return grouped_df



def get_map_df(df_with_countries):
    """
    Prepares the dataframe of movie plots with countries to be plotted in a map
     by decenny and dominant emotion
    """
    df_map = average_emotions_per_country(df_with_countries)
    df_map = get_most_dominant_emotion(df_map)
    
    #Make 'decenny' in a chronological order
    df_map['decenny'] = pd.Categorical(
        df_map['decenny'],
        categories=sorted(df_map['decenny'].unique()),
        ordered=True
    )
    
    #Sort the dataframe by decenny
    df_map = df_map.sort_values(by='decenny')
    df_map['Color'] = df_map['dominant_emotion'].map(COLORS)


    column_rename_map = {
        'normalized_plot_anger_without_neutral': 'Anger',
        'normalized_plot_disgust_without_neutral': 'Disgust',
        'normalized_plot_fear_without_neutral': 'Fear',
        'normalized_plot_joy_without_neutral': 'Joy',
        'normalized_plot_sadness_without_neutral': 'Sadness',
        'normalized_plot_surprise_without_neutral': 'Surprise'
    }

    df_map = df_map.rename(columns=column_rename_map)
    
    #Convert emotions scores to percentages
    for col in column_rename_map.values():
        df_map[col] = (df_map[col] * 100).round(2).astype(str) + '%'

    #to store sorted emotion scores
    emotion_columns = ['Anger', 'Disgust', 'Fear', 'Joy', 'Sadness', 'Surprise']

    #Function to get sorted emotion values in descending order
    def get_sorted_emotions(row):
        emotion_values = {emotion: row[emotion] for emotion in emotion_columns}
        sorted_emotions = sorted(emotion_values.items(), key=lambda x: float(x[1].replace('%', '')), reverse=True)
        #<br> so that each one is displayed in a new line when hovered
        return '<br>'.join([f"{emotion}: {score}" for emotion, score in sorted_emotions])

    df_map['Sorted Emotions'] = df_map.apply(get_sorted_emotions, axis=1)
    
    return df_map


def get_map_reviews(df_with_countries):
    """
    Prepares the dataframe of reviews with countries to be plotted in a map
     by decenny and dominant emotion
    """
    df_map = average_emotions_per_country(df_with_countries, reviews=True)
    df_map = get_most_dominant_emotion(df_map, reviews=True)
    
    #Make 'decenny' in a chronological order
    df_map['decenny'] = pd.Categorical(
        df_map['decenny'],
        categories=sorted(df_map['decenny'].unique()),
        ordered=True
    )
    
    #Sort the dataframe by decenny
    df_map = df_map.sort_values(by='decenny')
    df_map['Color'] = df_map['dominant_emotion'].map(COLORS)

    #Rename columns for better display in the map
    column_rename_map = {
        'review_anger' : 'Anger',
        'review_disgust' : 'Disgust',
        'review_fear' : 'Fear',
        'review_joy' : 'Joy',
        'review_sadness' : 'Sadness',
        'review_surprise' : 'Surprise'
    }

    df_map = df_map.rename(columns=column_rename_map)
    
    #Convert emotions scores to percentages
    for col in column_rename_map.values():
        df_map[col] = (df_map[col] * 100).round(2).astype(str) + '%'

    #to store sorted emotion scores
    emotion_columns = ['Anger', 'Disgust', 'Fear', 'Joy', 'Sadness', 'Surprise']

    #Function to get sorted emotion values in descending order
    def get_sorted_emotions(row):
        emotion_values = {emotion: row[emotion] for emotion in emotion_columns}
        sorted_emotions = sorted(emotion_values.items(), key=lambda x: float(x[1].replace('%', '')), reverse=True)
        #<br> so that each one is displayed in a new line when hovered
        return '<br>'.join([f"{emotion}: {score}" for emotion, score in sorted_emotions])

    df_map['Sorted Emotions'] = df_map.apply(get_sorted_emotions, axis=1)
    
    return df_map


def get_rating_and_plots(df_with_countries):
    """
    Returns the average rating and number of plots per country
    """
    df_with_countries['Country'] = df_with_countries.apply(get_accurate_countries, axis=1)
    nb_plots_per_country = df_with_countries.groupby('Country').size().reset_index(name='Plots Number')
    avg_rating_per_country = df_with_countries.groupby('Country')['averageRating'].mean().reset_index()
    merged_df = pd.merge(nb_plots_per_country, avg_rating_per_country, on='Country')
    merged_df.rename(columns={'averageRating': 'Average Rating'}, inplace=True)
    merged_df['Average Rating'] = merged_df['Average Rating'].round(2)
    return merged_df

