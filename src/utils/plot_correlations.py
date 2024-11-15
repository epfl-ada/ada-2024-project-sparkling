import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import pearsonr

def plot_top_country_emotion_correlation(df_emotions, df_country):

    """
    Plots the top 10 positive and negative correlations between emotions and countries.
    This function merges emotion and country data on 'wikipedia_ID' to ensure each row 
    represents a movie with its respective country and emotion scores. We calculate the 
    correlation with pearson correlation and identify the top 10 strongest positive and 
    negative correlations between each emotion and countries.

    Parameters:
    - df_emotions (pd.DataFrame): DataFrame containing emotion (without neutral) scores with 'wikipedia_ID'.
    - df_country (pd.DataFrame): DataFrame containing country information with 'wikipedia_ID'.

    Returns:
    - Bar plots for each emotion with the top positive and negative correlations by country.
    """

    emotion_labels = ["anger", "disgust", "fear", "joy", "sadness", "surprise"]
    emotion_columns = [f'normalized_plot_{emotion}_without_neutral' for emotion in emotion_labels]
    
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
        sns.barplot(x=top_countries.index, y=top_countries.values, palette="coolwarm_r", legend=False, hue=top_countries.index)
        plt.title(f"Top 10 positive and negative country correlations with {emotion.replace('normalized_plot_', '').replace('_without_neutral', '')}")
        plt.xlabel("Country")
        plt.ylabel("Correlation")
        plt.xticks(rotation=45, ha="right")
        plt.tight_layout()
        plt.show()

def generate_emotion_genre_heatmap(df_genres, df_emotions):

    """
    Generates a heatmap showing the correlation and p-values between emotions and genres.
    This function merges two DataFrames (one with genres and one with emotions) based on movie IDs so that each 
    row represents a single movie with both its genre and emotion data, to calculates the correlation between 
    emotions and genres with pearson correlation method, and their p-values.

    Parameters:
    - df_genres (pd.DataFrame): DataFrame with binary columns representing different genres.
    - df_emotions (pd.DataFrame): DataFrame containing columns for normalized emotion (without neutral) scores per movie.

    Returns:
    - A heatmap of the correlations and p-values.
    """
    
    genre_columns = df_genres.columns.difference(['wikipedia_ID'])
    emotion_labels = ["anger", "disgust", "fear", "joy", "sadness", "surprise"]
    emotion_columns = [f'normalized_plot_{emotion}_without_neutral' for emotion in emotion_labels]

    #To have in each line the emotions and genres corresponding to the same movie
    df_with_emotions_and_genres = pd.merge(df_genres, df_emotions, on='wikipedia_ID', how='inner')

    #Initialize matrices for correlations and p-values
    correlation_matrix = pd.DataFrame(index=emotion_columns, columns=genre_columns)
    p_value_matrix = pd.DataFrame(index=emotion_columns, columns=genre_columns)

    #Calculate correlations and p-values
    for emotion in emotion_columns:
        for genre in genre_columns:
            corr, p_value = pearsonr(
                df_with_emotions_and_genres[emotion], 
                df_with_emotions_and_genres[genre].astype(float)
            )
            correlation_matrix.loc[emotion, genre] = corr
            p_value_matrix.loc[emotion, genre] = p_value

    #Convert matrices to float for plotting
    correlation_matrix = correlation_matrix.astype(float)
    p_value_matrix = p_value_matrix.astype(float)

    #Plotting the correlation heatmap
    plt.figure(figsize=(12, 8))
    sns.heatmap(correlation_matrix, cmap="coolwarm", center=0, cbar=True, annot=True, fmt=".2f",
                annot_kws={"size": 7}, linewidths=0.5, yticklabels=emotion_labels)
    plt.xticks(rotation=45, ha='right', fontsize=9)
    plt.yticks(rotation=0, fontsize=9)
    plt.title('Correlation between emotions and genres', fontsize=14)
    plt.tight_layout()
    plt.show()

    #Plotting the p-value heatmap
    plt.figure(figsize=(12, 8))
    sns.heatmap(p_value_matrix, cmap="Greens_r", center=0, cbar=True, annot=True, fmt=".1e",
                annot_kws={"size": 6}, linewidths=0.5, yticklabels=emotion_labels, vmax=0.05)
    plt.xticks(rotation=45, ha='right', fontsize=9)
    plt.yticks(rotation=0, fontsize=9)
    plt.title('P-values between emotions and genres', fontsize=14)
    plt.tight_layout()
    plt.show()