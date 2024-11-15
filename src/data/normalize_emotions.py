import pandas as pd

def normalize_total_plot_emotions(df_movies, with_neutral=False):
    """
    Normalize plot emotion scores in a DataFrame, with or without the 'neutral' emotion.

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
    
    # Define emotion columns for normalization
    base_emotion_columns = ['plot_anger', 'plot_disgust', 'plot_fear', 'plot_joy', 'plot_sadness', 'plot_surprise']
    all_emotion_columns = base_emotion_columns + ['plot_neutral']
    
    # Determine which columns to use based on with_neutral flag
    selected_columns = all_emotion_columns if with_neutral else base_emotion_columns
    
    # Compute the sum of each list in the columns
    for col in selected_columns:
        df_movies[f'sum_{col}'] = df_movies[col].apply(sum)
    
    # Calculate the total sum of emotions for each row
    total_emotion_sum = df_movies[[f'sum_{col}' for col in selected_columns]].sum(axis=1)
    
    # Normalize each summed emotion score by dividing by the total sum of emotions
    for col in selected_columns:
        suffix = 'with_neutral' if with_neutral else 'without_neutral'
        df_movies[f'normalized_{col}_{suffix}'] = df_movies[f'sum_{col}'] / total_emotion_sum
        
    # Delete the intermediate sum columns
    for col in selected_columns:
        df_movies = df_movies.drop([f'sum_{col}'], axis=1)
    
    # Return the updated DataFrame with normalized emotion columns
    return df_movies


def normalize_review_emotions(df_reviews, with_neutral=False):
    """
    Normalize review emotion scores in a DataFrame, with or without the 'neutral' emotion.

    This function takes a DataFrame with columns representing different emotions 
    (anger, disgust, fear, joy, sadness, surprise, and optionally neutral), where each cell 
    contains a single float value. It normalizes each score across the specified emotion 
    columns for each row.

    Parameters:
    - df_reviews (pd.DataFrame): DataFrame containing emotion scores for each review.
    - with_neutral (bool): If True, includes the 'review_neutral' column in the normalization.

    Returns:
    - pd.DataFrame: DataFrame with additional columns for normalized emotion scores.
    """
    
    # Work on a copy
    df_reviews = df_reviews.copy()
    
    # Define emotion columns for normalization
    base_emotion_columns = ['review_anger', 'review_disgust', 'review_fear', 'review_joy', 'review_sadness', 'review_surprise']
    all_emotion_columns = base_emotion_columns + ['review_neutral']
    
    # Determine which columns to use based on with_neutral flag
    selected_columns = all_emotion_columns if with_neutral else base_emotion_columns
    
    # Calculate the total sum of emotion scores for each row
    total_emotion_sum = df_reviews[selected_columns].sum(axis=1)
    
    # Normalize each emotion score by dividing by the total sum of emotions
    for col in selected_columns:
        suffix = 'with_neutral' if with_neutral else 'without_neutral'
        df_reviews[f'normalized_{col}_{suffix}'] = df_reviews[col] / total_emotion_sum
    
    # Return the updated DataFrame with normalized emotion columns
    return df_reviews
