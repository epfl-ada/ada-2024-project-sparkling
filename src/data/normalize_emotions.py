import pandas as pd

def normalize_total_plot_emotions(df_movies, with_neutral=False):
    """
    Normalize plot emotion scores in a DataFrame.

    This function takes a DataFrame with columns representing different emotions 
    (anger, disgust, fear, joy, sadness, surprise, and optionally neutral). Each cell contains 
    a list of floats, and the function first sums these lists, then normalizes the summed 
    values across the specified emotion columns for each row.

    Parameters:
    - df_movies (pd.DataFrame): DataFrame containing lists of emotion scores for each plot.
    - with_neutral (bool): If True, includes the 'plot_neutral' column in the normalization.

    Returns:
    - pd.DataFrame: DataFrame with additional columns for normalized summed emotion scores.
    """
    
    # Work on a copy
    df_movies = df_movies.copy()
    
    # Define the columns to include in the normalization process
    emotion_columns = ['plot_anger', 'plot_disgust', 'plot_fear', 'plot_joy', 'plot_sadness', 'plot_surprise']
    
    # Include 'plot_neutral' if specified
    if with_neutral:
        emotion_columns.append('plot_neutral')
    
    # Compute the sum of each list in the columns
    for col in emotion_columns:
        df_movies[f'sum_{col}'] = df_movies[col].apply(sum)
    
    # Calculate the total sum of emotions for each row
    total_emotion_sum = df_movies[[f'sum_{col}' for col in emotion_columns]].sum(axis=1)
    
    # Normalize each summed emotion score by dividing by the total sum of emotions
    for col in emotion_columns:
        df_movies[f'normalized_{col}'] = df_movies[f'sum_{col}'] / total_emotion_sum
        
    # Delete the intermediate sum columns
    for col in emotion_columns:
        df_movies = df_movies.drop([f'sum_{col}'], axis=1)
    
    # Return the updated DataFrame with normalized emotion columns
    return df_movies


def normalize_review_emotions(df_reviews, with_neutral=False):
    """
    Normalize review emotion scores in a DataFrame.

    This function takes a DataFrame with columns representing different emotions 
    (anger, disgust, fear, joy, sadness, surprise, and optionally neutral), where each cell 
    contains a single float value. It normalizes each score across the specified emotion 
    columns for each row.

    Parameters:
    - df_movies (pd.DataFrame): DataFrame containing emotion scores for each review.
    - with_neutral (bool): If True, includes the 'review_neutral' column in the normalization.

    Returns:
    - pd.DataFrame: DataFrame with additional columns for normalized emotion scores.
    """
    
    # Work on a copy
    df_reviews = df_reviews.copy()
    
    # Define the columns to include in the normalization process
    emotion_columns = ['review_anger', 'review_disgust', 'review_fear', 'review_joy', 'review_sadness', 'review_surprise']
    
    # Include 'review_neutral' if specified
    if with_neutral:
        emotion_columns.append('review_neutral')
    
    # Calculate the total sum of emotion scores for each row
    total_emotion_sum = df_reviews[emotion_columns].sum(axis=1)
    
    # Normalize each emotion score by dividing by the total sum of emotions
    for col in emotion_columns:
        df_reviews[f'normalized_{col}'] = df_reviews[col] / total_emotion_sum
    
    # Return the updated DataFrame with normalized emotion columns
    return df_reviews