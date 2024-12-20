import pandas as pd
import numpy as np
from scipy.stats import pearsonr
import plotly.graph_objects as go
import plotly.express as px
from src.utils.initial_plots_data import COLORS, EMOTIONS
from src.utils.save_plot import save_plot

# List of movie genres
GENRES = ['Action', 'Adventure', 'Animated', 'Comedy', 'Crime', 'Disaster', 
         'Documentary', 'Drama', 'Experimental', 'Fantasy', 'Historical', 
         'Horror', 'Musical', 'Noir', 'Pornographic', 'Propaganda', 'Romance', 
         'Science Fiction', 'Teen', 'Thriller', 'War', 'Western']


def generate_emotion_genre_heatmap(df_genres, df_emotions):
    """
    Generates interactive heatmaps showing correlations and p-values between emotions and genres

    Parameters:
    - df_genres (pd.DataFrame): Dataframe with binary columns representing different genres
    - df_emotions (pd.DataFrame): Dataframe containing columns for normalized emotion scores per movie

    Returns:
    - None: The function directly generates and displays a plotly figure

    Outputs:
    - Two interactive heatmaps: one for correlations and one for p-values
    """
    # Define genres and emotions
    genre_columns = df_genres.columns.difference(['wikipedia_ID'])
    emotion_labels = ["Anger", "Disgust", "Fear", "Joy", "Sadness", "Surprise"]
    emotion_columns = [f'normalized_plot_{emotion.lower()}_without_neutral' for emotion in emotion_labels]

    # Merge the dataframes to align genres and emotions by movie
    df_with_emotions_and_genres = pd.merge(df_genres, df_emotions, on='wikipedia_ID', how='inner')

    # Initialize matrices for correlations and p-values
    correlation_matrix = pd.DataFrame(index=emotion_columns, columns=genre_columns)
    p_value_matrix = pd.DataFrame(index=emotion_columns, columns=genre_columns)

    # Calculate correlations and p-values
    for emotion in emotion_columns:
        for genre in genre_columns:
            corr, p_value = pearsonr(
                df_with_emotions_and_genres[emotion],
                df_with_emotions_and_genres[genre].astype(float)
            )
            correlation_matrix.loc[emotion, genre] = corr
            p_value_matrix.loc[emotion, genre] = p_value

    # Convert to numeric for plotting
    correlation_matrix = correlation_matrix.astype(float)
    p_value_matrix = p_value_matrix.astype(float)

    # Plot correlation heatmap with text in each cell
    fig_corr = go.Figure(
        data=go.Heatmap(
            z=correlation_matrix.values.T,
            x=emotion_labels,
            y=genre_columns,
            colorscale='RdBu_r',  
            zmid=0,
            text=correlation_matrix.T.map(lambda x: f"{x:.2f}").values,
            hovertemplate="Correlation [%{x} - %{y}]: %{z:.2f}<extra></extra>", 
            showscale=True,
            texttemplate="%{text}", 
            textfont={"size": 10}  
        )
    )
    
    fig_corr.update_layout(
        title="Correlation between genres and emotions",
        xaxis=dict(title="Emotions", tickangle=45, showgrid=True),
        yaxis=dict(title="Genres", showgrid=True),
        template="plotly_white",
        width= 800,
        height = 600,
    )

    # Plot p-value heatmap with text in each cell
    fig_pval = go.Figure(
        data=go.Heatmap(
            z=p_value_matrix.values.T,
            x=emotion_labels,
            y=genre_columns,
            colorscale=[
               [0.0, 'lightgreen'],   
               [1.0, 'white'] 
            ],
            zmin=0,
            zmax=0.8,
            text=p_value_matrix.T.map(lambda x: f"{x:.1e}").values,
            hovertemplate="P-value [%{x} - %{y}]: %{z:.1e}<extra></extra>", 
            showscale=True,
            texttemplate="%{text}", 
            textfont={"size": 9},
        )
    )

    
    fig_pval.update_layout(
        title="P-values between genres and emotions<br>H0: No significant relationship between the emotion and the genre<br>p_value < 0.05 : correlation coefficient statistically significant",
        xaxis=dict(
            title="Emotions",
            tickangle=45,  
        ),
        yaxis=dict(
            title="Genres",  
        ),
        template="plotly_white",
        width= 800,
        height = 600,
    )

    # Show heatmaps
    fig_corr.show()
    fig_pval.show()
    save_plot(fig_corr, "emotion_genre_heatmap_corr")
    save_plot(fig_pval, "emotion_genre_heatmap_p_val")

    

def plot_genres_proportions_sorted(df_genres, show_legend=True):
    
    """
    This function calculates the proportion of movies for each genre from the provided dataframe, 
    sorts the genres in ascending order based on their counts, and visualizes the results in a pie chart.
    Each slice of the chart represents a genre, annotated with its percentage and name

    Parameters:
    - df_genres (pd.DataFrame): A dataframe containing one-hot encoded columns for genres
    - show_legend (bool): If true, displays the legend on the chart

    Returns:
    - None: The function directly generates and displays a plotly pie chart

    Outputs:
    - A pie chart showing:
        Proportions of movies belonging to each genre
        Labels with genre names and percentages inside the chart
    """

    # Calculation of proportions and sorting them in ascending order
    genres = df_genres.drop(columns=["wikipedia_ID"], errors="ignore")
    genres_counts = genres.sum().sort_values()  
    proportions = genres_counts / genres_counts.sum() * 100

    # Plotting the genres with their %
    fig = go.Figure(data=[
        go.Pie(
            labels=genres_counts.index,
            values=genres_counts.values,
            textinfo='percent+label',  
            textposition='inside',  
            hoverinfo='label+value+percent',
            marker=dict(colors=px.colors.qualitative.Set3),  
            sort=False  
        )
    ])

    # Axis and legend styling
    fig.update_layout(
        title="Proportion of movies genres",
        title_font_size=20,
        legend_title=dict(text="Genres <br>") if show_legend else None,  
        showlegend=show_legend,
        width= 700,
        height = 700,
    )

    save_plot(fig, f"genres_proportions_{show_legend}.html")

    fig.show()


def plot_emotion_distribution(df_with_emotions_normalized, df_main_genres, is_review=False, specific_emotion=None):
    """
    Generates a stacked bar chart to visualize the distribution of emotions across movie genres.

    Args:
        df_with_emotions_normalized (pd.DataFrame): DataFrame containing normalized emotion scores for movies,
            with each emotion as a column.
        df_main_genres (pd.DataFrame): DataFrame with binary genre labels for movies. Rows correspond to movies,
            and columns represent genres.
        is_review (bool, optional): Indicates whether emotion scores are derived from reviews (True) or plots (False).
            Defaults to False.
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

    if specific_emotion is not None and not is_review:
        fig.update_layout(
            yaxis=dict(
                range=[0, 30],
            ),
        )

    if specific_emotion is not None and is_review:
        fig.update_layout(
            yaxis=dict(
                range=[0, 45],
            ),
        )


    save_plot(fig, f"emotion_distribution_of_movie_{'reviews' if is_review else 'plots'}_across_genres")

    # Force the width and the height for the notebook
    fig.update_layout(
        width= 800,
        height = 400,
    )
    fig.show()



