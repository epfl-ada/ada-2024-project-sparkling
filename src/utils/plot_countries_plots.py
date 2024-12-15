import pandas as pd
import plotly.express as px
import numpy as np

from src.utils.initial_plots_data import COLORS
from src.utils.save_plot import save_plot


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



def plot_world_map_emotion_by_genre(df_emotions_normalized, df_countries, is_reviews):
    """
    Given the dataframe containing emotions of the movie's plot/reviews normalized without the neutral emotion and the dataframe containing
    the countries associated to the movies. Plot a world map displaying 
    the most dominant emotions per countries over multiple decennies

    Argument:
        - df_movies_with_emotions_normalized: Dataframe containing emotions of the movie's plot or reviews normalized without the neutral emotion
        - df_countries: Dataframe containing the countries associated to the movies
        - is_reviews: Whether the df_movies_with_emotions_normalized dataframe contains reviews emotions
    """

    df_with_countries = get_decenny_and_country(df_emotions_normalized, df_countries)
    if not is_reviews:
        df_map = get_map_df(df_with_countries)
    else:
        df_map = get_map_reviews(df_with_countries)

    # We need to have at least one country with each emotion to be able to show it
    # This is a work around because it seems that to have the legend to show a particular emotion (and also show the emotion in the map)
    # on the frame, the legend has to display it. So if the first frame does not have all the emotions
    # shown, then the emotion will not be shown for any decenny.
    # Work around: Add a fake country that have all the emotions at the first decenny
    # Moreover: After testing it seems that when we move over the decenny, if one emotion is not present, then it may be replaced by another
    # For example have two "surprise" emotion in the legend
    emotions = COLORS.keys()
    nb_emotions = len(emotions)
    list_dataframe_decennies = []
    for decenny in df_map["decenny"].unique():
        fake_decenny_countries = {"Country": ["fake_country"] * nb_emotions, "dominant_emotion": emotions, "decenny": [decenny] * nb_emotions}
        list_dataframe_decennies.append(pd.DataFrame(fake_decenny_countries))
    df_map = pd.concat([df_map] + list_dataframe_decennies)


    fig = px.choropleth(
        data_frame=df_map,
        locations="Country",           
        locationmode="country names",  
        color="dominant_emotion",
        color_discrete_map=COLORS,     
        animation_frame="decenny",     #temporal slider
        title="Emotion by country over time",
        projection="natural earth",
        hover_data={'Country': True, 'Sorted Emotions': True, 'Anger': False, 'decenny': False,
                    'Disgust': False, 'Fear': False, 'Joy': False, 'Sadness': False, 'Surprise': False, 'dominant_emotion': False}
    )


    # We had a hard time to show the hovertemplate as we wanted
    # So by printing fig, we found this method by changing the dictionnary of the figure
    fig.update_traces(hovertemplate='<b>%{customdata[0]}</b><br><br>%{customdata[1]}<extra></extra>')

    for frame in fig.frames:
        for graphs in frame.data:
            graphs.hovertemplate = graphs.hovertemplate.replace("Sorted Emotions=", "<br>").replace("Country=", "").replace("%{customdata[0]}", "<b>%{customdata[0]}</b>") + "<extra></extra>"

    fig.update_layout(
        geo=dict(showframe=False, showcoastlines=True),
        legend_title="Emotion",
        transition={'duration': 3000}  #transition duration to 3 seconds
    )

    fig.layout.updatemenus[0].buttons[0].args[1]['frame']['duration'] = 3000  #Frame duration: 3 seconds
    fig.layout.updatemenus[0].buttons[0].args[1]['transition']['duration'] = 3000  #Transition duration: 3 seconds

    fig.show()

    save_plot(fig, ("plot" if not is_reviews else "reviews")+"_map_final")


def plot_world_map_average_rating(df_movies_with_emotions_normalized, df_countries):
    """
    Given the dataframe containing emotions of the movie's plot normalized without the neutral emotion and the dataframe containing
    the countries associated to the movies. Plot a world map displaying 
    the average ratings per countries

    Argument:
        - df_movies_with_emotions_normalized: Dataframe containing emotions of the movie's plot normalized without the neutral emotion
        - df_countries: Dataframe containing the countries associated to the movies
    """

    #scaling factor for bubble sizes
    scaling_factor = 8  

    df_with_countries = get_decenny_and_country(df_movies_with_emotions_normalized, df_countries)
    ratings_df = get_rating_and_plots(df_with_countries)

    ratings_df["Log Plots Number"] = np.log1p(ratings_df["Plots Number"])  # Using log1p to handle zero values safely

    fig = px.scatter_geo(
        ratings_df,
        locations="Country",
        locationmode="country names",
        size="Log Plots Number",  # Bubble size based on log of plots number
        color="Average Rating",  # Color based on average rating
        hover_name="Country",  # Hover text showing country name
        hover_data={"Plots Number": True, "Log Plots Number": False},  # Show original plots number, hide log value
        title="Countries average ratings",
        projection="natural earth", 
        color_continuous_scale=px.colors.sequential.Viridis,
    )

    # Bubble size scaling using logarithmic sizes
    fig.update_traces(
        marker=dict(
            sizemode='diameter',
            sizeref=2. * max(ratings_df["Log Plots Number"]) / (scaling_factor ** 2),
            sizemin=5  # Minimum bubble size for visibility
        )
    )

    fig.show()

    save_plot(fig, "ratings_map_final")
