import plotly.graph_objects as go
import pandas as pd
import os

# List of emotions
EMOTIONS = ["anger", "disgust", "fear", "joy", "sadness", "surprise"]

# List of movie genres
GENRES = ['Action', 'Adventure', 'Animated', 'Comedy', 'Crime', 'Disaster', 
         'Documentary', 'Drama', 'Experimental', 'Fantasy', 'Historical', 
         'Horror', 'Musical', 'Noir', 'Pornographic', 'Propaganda', 'Romance', 
         'Science Fiction', 'Teen', 'Thriller', 'War', 'Western']

# Dictionary of colors for each emotion
COLORS = {
    'joy': '#f3c35f',
    'anger': '#e13c29',
    'disgust': '#8bc664',
    'sadness': '#85b8ed',
    'fear': '#b99be5',
    'surprise': '#de9f81'
}

PLOTS_PATH = 'plots'



def plot_emotion_distribution(df_with_emotions_normalized, df_main_genres, is_review=False, filename='plot', specific_emotion=None):
    """
    Generates a stacked bar chart to visualize the distribution of emotions across movie genres.

    Args:
        df_with_emotions_normalized (pd.DataFrame): DataFrame containing normalized emotion scores for movies,
            with each emotion as a column.
        df_main_genres (pd.DataFrame): DataFrame with binary genre labels for movies. Rows correspond to movies,
            and columns represent genres.
        is_review (bool, optional): Indicates whether emotion scores are derived from reviews (True) or plots (False).
            Defaults to False.
        filename (str, optional): Name of the output HTML file (without extension) where the plot will be saved.
            Defaults to 'plot'.
        specific_emotion (str, optional): If specified, highlights only the provided emotion in the plot,
            hiding others. Should match the base emotion name (e.g., "joy", "anger"). Defaults to None.

    Returns:
        None: 
            - Displays the generated interactive chart in the browser.
            - Saves the chart as an HTML file in the predefined output directory.
    """
    # Merge emotion scores with genre data on 'wikipedia_ID'
    df_emotions_with_genres = df_with_emotions_normalized.merge(
        df_main_genres, 
        on='wikipedia_ID'
    )

    # Define emotion column names dynamically based on the data source (review or plot)
    EMOTION_COLUMNS = [f'normalized_{"review" if is_review else "plot"}_{emotion}_without_neutral' for emotion in EMOTIONS]

    # Calculate the emotion distribution for each genre
    genre_emotion_data = {}
    for genre in GENRES:
        # Filter rows corresponding to the current genre and sum their emotion scores
        genre_data = df_emotions_with_genres[df_emotions_with_genres[genre]].loc[:, EMOTION_COLUMNS].sum()
        # Normalize the emotion scores to percentages
        genre_emotion_data[genre] = genre_data / genre_data.sum() * 100

    # Convert the aggregated data to a DataFrame for easier manipulation
    emotion_df = pd.DataFrame(genre_emotion_data).T  # Transpose to have genres as rows

    # Create a Plotly Figure for the stacked bar chart
    fig = go.Figure()

    for emotion in EMOTION_COLUMNS:
        # Extract the base emotion name from the column
        emotion_name = emotion.split('_')[2]
        # Add a trace for each emotion to the bar chart
        fig.add_trace(
            go.Bar(
                name=emotion_name.capitalize(),  # Display emotion name in capitalized format
                x=emotion_df.index,  # Genre names on the x-axis
                y=emotion_df[emotion],  # Emotion scores on the y-axis
                marker_color=COLORS.get(emotion_name, '#000000'),  # Use defined colors or default to black
                hovertemplate=(
                    "<b>Genre:</b> %{x}<br>"
                    "<b>Emotion:</b> " + emotion_name + "<br>"
                    "<b>Score:</b> %{y:.2f}%<extra></extra>"  # Hover tooltip
                ),
                visible='legendonly' if specific_emotion is not None and emotion_name != specific_emotion else True  # specific_emotion is ticked by default
            )
        )

    # Configure the chart layout
    fig.update_layout(
        barmode="stack",  # Stack bars for cumulative representation
        title=f"Emotion Distribution of Movie {'Reviews' if is_review else 'Plots'} Across Genres",
        xaxis_title="Genre",
        yaxis_title="Emotion Score (%)",
        legend_title="Emotion",
        title_x=0.5,
        title_xanchor="center",
        xaxis=dict(type='category', categoryorder='total descending'),  # Order genres by total emotion scores
        template="plotly_white"  # Use a clean background style
    )

    # Ensure the output directory exists
    if not os.path.exists(PLOTS_PATH):
        os.makedirs(PLOTS_PATH)

    # Save the plot as an interactive HTML file
    filepath = os.path.join(PLOTS_PATH, f"{filename}.html")
    fig.write_html(filepath)

    # Display the chart in the browser
    fig.show()



