import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

def generate_emotion_genre_heatmap(df_genres, df_emotions):

    """
    Generates a heatmap showing the correlation between emotions and genres.
    This function merges two DataFrames (one with genres and one with emotions) based on movie IDs so that each 
    row represents a single movie with both its genre and emotion data, to calculates the correlation between 
    emotions and genres with pearson correlation method.

    Parameters:
    - df_genres (pd.DataFrame): DataFrame with binary columns representing different genres.
    - df_emotions (pd.DataFrame): DataFrame containing columns for normalized emotion scores per movie.

    Returns:
    - A heatmap of the correlation matrix.
    """
    
    genre_columns = df_genres.columns.difference(['wikipedia_ID'])
    emotion_columns = ['normalized_plot_anger', 'normalized_plot_disgust', 'normalized_plot_fear', 
                       'normalized_plot_joy', 'normalized_plot_sadness', 'normalized_plot_surprise']

    #To have in each line the emotions and genres corresponding to the same movie
    df_with_emotions_and_genres = pd.merge(df_genres, df_emotions, on='wikipedia_ID', how='inner')

    #Calcultion avec the correlation matrix
    correlation_matrix = pd.concat([df_with_emotions_and_genres[emotion_columns], 
                                    df_with_emotions_and_genres[genre_columns].astype(float)], 
                                    axis=1).corr(method='pearson').loc[emotion_columns, genre_columns] 

    #Plotting the data 
    emotion_labels = [emotion.replace('normalized_plot_', '') for emotion in emotion_columns]
    plt.figure(figsize=(12, 8))
    sns.heatmap(correlation_matrix, cmap="coolwarm", center=0, cbar=True, annot=True, fmt=".2f",
                annot_kws={"size": 7}, linewidths=0.5, yticklabels=emotion_labels)
    plt.xticks(rotation=45, ha='right', fontsize=9)
    plt.yticks(rotation=0, fontsize=9)
    plt.title('Correlation between emotions and genres', fontsize=14)
    plt.tight_layout()
    plt.show()

def plot_top_country_emotion_correlation(df_emotions, df_country):

    """
    Plots the top 10 positive and negative correlations between emotions and countries.
    This function merges emotion and country data on 'wikipedia_ID' to ensure each row 
    represents a movie with its respective country and emotion scores. We calculate the 
    correlation with pearson correlation and identify the top 10 strongest positive and 
    negative correlations between each emotion and countries.

    Parameters:
    - df_emotions (pd.DataFrame): DataFrame containing emotion scores with 'wikipedia_ID'.
    - df_country (pd.DataFrame): DataFrame containing country information with 'wikipedia_ID'.

    Returns:
    - Bar plots for each emotion with the top positive and negative correlations by country.
    """
    
    emotion_columns = ['normalized_plot_anger', 'normalized_plot_disgust', 'normalized_plot_fear', 
                       'normalized_plot_joy', 'normalized_plot_sadness', 'normalized_plot_surprise']
    
    #To have in each line the emotions and country corresponding to the same movie
    df_with_emotions_and_contries = pd.merge(df_emotions, df_country, on='wikipedia_ID', how='inner')

    #Make a list of the different countries
    country_columns = df_country.columns.difference(['wikipedia_ID'])

    #Calcultion avec the correlation matrix
    correlation_matrix = pd.concat([df_with_emotions_and_contries[emotion_columns], 
                                    df_with_emotions_and_contries[country_columns]], 
                                    axis=1).corr(method='pearson').loc[emotion_columns, country_columns]

    #Create a graph for each emotion
    for emotion in emotion_columns:
        #Obtain the top 10 countries with the strongest positive and negative correlations for each emotion
        top_positive = correlation_matrix.loc[emotion].nlargest(10)
        top_negative = correlation_matrix.loc[emotion].nsmallest(10)
        top_countries = pd.concat([top_positive, top_negative])

        plt.figure(figsize=(12, 8))
        sns.barplot(x=top_countries.index, y=top_countries.values, palette="coolwarm", legend=False, hue=top_countries.index)
        plt.title(f"Top 10 positive and negative country correlations with {emotion.replace('normalized_plot_', '')}")
        plt.xlabel("Country")
        plt.ylabel("Correlation")
        plt.xticks(rotation=45, ha="right")
        plt.tight_layout()
        plt.show()

def plot_emotion_distribution_per_period(dataframe, period_column):

    """
    Generates a stacked bar chart of emotion distribution for each period : in our case we use it for months ans years.
    For each period (month or year), we split a 1-unit bar into segments representing the proportion of each emotion 
    within that period.

    Parameters:
    - dataframe (pd.DataFrame): DataFrame with emotion scores and specified period column.
    - period_column (str): Column name for grouping data ('release_year' or 'release_month').

    Returns:
    - Stacked bar chart showing the distribution of each emotion per period.
    """
    
    emotions = ['normalized_plot_anger', 'normalized_plot_disgust', 'normalized_plot_fear', 
                'normalized_plot_joy', 'normalized_plot_sadness', 'normalized_plot_surprise']
    
    # calculate the proportion of each emotion for each period
    period_emotion_totals = dataframe.groupby(period_column)[emotions].sum()
    period_emotion_distribution = period_emotion_totals.div(period_emotion_totals.sum(axis=1), axis=0)
    
    # Create the plot
    plt.figure(figsize=(14, 8))
    bottom = None
    colors = sns.color_palette("Dark2", len(emotions))  

    #One bar for each emotion
    for i, emotion in enumerate(emotions):
        emotion_label = emotion.replace("normalized_plot_", "")
        if bottom is None:
            bottom = period_emotion_distribution[emotion]
            plt.bar(period_emotion_distribution.index, period_emotion_distribution[emotion], 
                    label=emotion_label, color=colors[i])
        else:
            plt.bar(period_emotion_distribution.index, period_emotion_distribution[emotion], bottom=bottom, 
                    label=emotion_label, color=colors[i])
            bottom += period_emotion_distribution[emotion]

    plt.title(f"Distribution of emotions per {period_column.replace('release_', '')}")
    plt.xlabel(period_column.replace('release_', '').capitalize())
    plt.ylabel("Proportion")
    plt.legend(title="Emotion", bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.show()

def calculate_deviation(dataframe, reference_period='release_month'):

    """
    Plots the deviation of emotions from their overall mean across the specified period to observe fluctuations or trends 
    in emotion intensities over time. This visualization helps to identify periods where certain emotions are notably higher 
    or lower than their overall average, providing insights into potential seasonal variations in emotional content.
    
    Parameters:
    - dataframe (pd.DataFrame): DataFrame containing emotion scores and the specified period column.
    - reference_period (str): The column name representing the period to group by ('release_month' or 'release_year').
    
    Returns:
    - Line plot showing the deviation of each emotion from its overall mean over the specified period.
    """
    
    emotions = ['normalized_plot_anger', 'normalized_plot_disgust', 'normalized_plot_fear', 
                'normalized_plot_joy', 'normalized_plot_sadness', 'normalized_plot_surprise']
    
    overall_mean = dataframe[emotions].mean()
    period_data = dataframe.groupby(reference_period)[emotions].mean()

    deviation_from_mean = period_data - overall_mean

    #Plotting deviations
    plt.figure(figsize=(12, 6))
    for emotion in emotions:
        emotion_label = emotion.replace("normalized_plot_", "")
        plt.plot(deviation_from_mean.index, deviation_from_mean[emotion], label=emotion_label)
        
    plt.title("Index of deviation of emotions from the mean")
    plt.xlabel(reference_period.replace('release_', '').capitalize())
    plt.ylabel("Deviation from the mean")
    plt.legend(title="Emotion", bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.show()

def plot_major_emotion_per_five_years(dataframe, year_column='release_year'):

    """
    This function groups data by five year intervals and finds the dominant emotion 
    (the one with the highest average score) in each period. It then visualizes the 
    percentage of the dominant emotion per period.

    Parameters:
    - dataframe (pd.DataFrame): DataFrame containing emotion scores and a release year column.
    - year_column (str): Column name for the release year (default is 'release_year').

    Returns:
    - Bar plot with the dominant emotion and its average percentage per period.
    """
    
    emotions = ['normalized_plot_anger', 'normalized_plot_disgust', 'normalized_plot_fear', 
                'normalized_plot_joy', 'normalized_plot_sadness', 'normalized_plot_surprise']
    
    #Group the data by five year intervals, using the beginning year of each period
    dataframe['period'] = (dataframe[year_column] // 5) * 5
    
    #Calculate the mean emotion scores for each period
    period_means = dataframe.groupby('period')[emotions].mean()
    
    #Find the dominant emotion and its percentage for each period
    dominant_emotion = period_means.idxmax(axis=1).str.replace('normalized_plot_', '')
    dominant_percentage = period_means.max(axis=1)
    
    #Add dominant emotion and percentage as new columns to period_means
    period_means['Dominant Emotion'] = dominant_emotion
    period_means['Percentage'] = dominant_percentage
    
    #Plotting
    plt.figure(figsize=(14, 6))
    sns.barplot(x=period_means.index, y='Percentage', hue='Dominant Emotion', data=period_means)
    plt.title("Dominant emotion for each quinquennium")
    plt.xlabel("Five year period")
    plt.ylabel("Proportion")
    plt.legend(title="Emotion", bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.tight_layout()
    plt.show()
