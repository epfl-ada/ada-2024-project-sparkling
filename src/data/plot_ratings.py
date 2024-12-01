import plotly.graph_objects as go
import pandas as pd
import os

from src.data.plot_genres import EMOTIONS, GENRES, COLORS, PLOTS_PATH



def emotion_distribution_by_movie_rating(df_movies_with_emotions_normalized, df_reviews_with_emotions_normalized=None, is_review=False, filename='plot', specific_emotion=None):
    """
    Generates a stacked bar chart visualizing the percentage distribution of emotions 
    across average movie rating bins (e.g., 0-1, 1-2, ..., 9-10).

    Args:
        df_movies_with_emotions_normalized (pd.DataFrame): DataFrame containing normalized emotion scores 
            derived from movie plots. Includes a column for average movie ratings ('averageRating').
        df_reviews_with_emotions_normalized (pd.DataFrame, optional): DataFrame containing normalized emotion scores 
            derived from movie reviews. Required if `is_review` is True. Defaults to None.
        is_review (bool, optional): Indicates whether to use review-based emotion scores (True) or plot-based scores (False). 
            Defaults to False.
        filename (str, optional): Name of the output HTML file (without extension) for saving the chart. Defaults to 'plot'.
        specific_emotion (str, optional): If specified, highlights only the provided emotion in the plot, hiding others. 
            Should match the base emotion name (e.g., "joy", "anger"). Defaults to None.

    Returns:
        None:
            - Displays the generated interactive chart in the browser.
            - Saves the chart as an HTML file in the predefined output directory.
    """
    # Merge the reviews data with the movie data if reviews are being analyzed
    if is_review:
        df = df_movies_with_emotions_normalized.merge(
            df_reviews_with_emotions_normalized, 
            on='wikipedia_ID'
        )
    else:
        df = df_movies_with_emotions_normalized

    # Create bins for average movie ratings, from 0-1, 1-2, ..., up to 9-10
    bins = range(0, 11)  # Define bins from 0 to 10 (exclusive of 10)
    labels = [f"{i}-{i+1}" for i in range(0, 10)]  # Bin labels (0-1, 1-2, ..., 9-10)
    
    # Assign each movie to its appropriate rating bin based on the 'averageRating' column
    df['rating_bin'] = pd.cut(
        df['averageRating'], 
        bins=bins, 
        labels=labels, 
        right=False  # Ensure the bins are left-inclusive
    )

    # Define the emotion columns dynamically based on the data source (review or plot)
    EMOTION_COLUMNS = [f'normalized_{"review" if is_review else "plot"}_{emotion}_without_neutral' for emotion in EMOTIONS]

    # Calculate the average emotion scores for each rating bin, then convert to percentages
    rating_emotion_data = df.groupby('rating_bin', observed=True)[EMOTION_COLUMNS].mean() * 100

    # Transpose the DataFrame for easier plotting (emotions as rows, rating bins as columns)
    emotion_df = rating_emotion_data.T

    # Initialize the Plotly figure for a stacked bar chart
    fig = go.Figure()

    for emotion in EMOTION_COLUMNS:
        # Extract the base emotion name from the column (e.g., 'happiness' from 'normalized_review_happiness')
        emotion_name = emotion.split('_')[2]
        
        # Add a bar trace for each emotion
        fig.add_trace(
            go.Bar(
                name=emotion_name.capitalize(),  # Capitalize the emotion name for display
                x=emotion_df.columns,  # Rating bins on the x-axis
                y=emotion_df.loc[emotion],  # Emotion score for each rating bin
                marker_color=COLORS.get(emotion_name, '#000000'),  # Use predefined colors or default to black
                hovertemplate=(
                    "<b>Rating Bin:</b> %{x}<br>"
                    "<b>Emotion:</b> " + emotion_name + "<br>"
                    "<b>Score:</b> %{y:.2f}%<extra></extra>"  # Display emotion score in percentage with 2 decimals
                ),
                visible='legendonly' if specific_emotion is not None and emotion_name != specific_emotion else True  # specific_emotion is ticked by default
            )
        )

    # Set the chart title based on the type of data (reviews or plots)
    title = f"Emotion Distribution of Movie {'Reviews' if is_review else 'Plots'} by Average Movie Rating Bins"

    # Update the layout for the stacked bar chart
    fig.update_layout(
        barmode="stack",  # Stack bars for a cumulative representation of emotions
        title=title,  # Set the title of the chart
        xaxis_title="Average Movie Rating Bin",  # Label for the x-axis
        yaxis_title="Emotion Score (%)",  # Label for the y-axis
        legend_title="Emotion",  # Legend title
        template="plotly_white",  # Use a clean white background style
    )

    # Ensure the output directory for plots exists
    if not os.path.exists(PLOTS_PATH):
        os.makedirs(PLOTS_PATH)

    # Save the chart as an interactive HTML file
    filepath = os.path.join(PLOTS_PATH, f"{filename}.html")
    fig.write_html(filepath)

    # Display the chart in the browser
    fig.show()



