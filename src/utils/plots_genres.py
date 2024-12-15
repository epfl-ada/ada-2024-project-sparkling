import pandas as pd
import numpy as np
from scipy.stats import pearsonr
import plotly.graph_objects as go
import plotly.express as px



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
    #pio.write_html(fig_corr, file="emotion_genre_heatmap_corr.html", auto_open=False, auto_play=False)
    #pio.write_html(fig_pval, file="emotion_genre_heatmap_p_val.html", auto_open=False, auto_play=False)

    

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
        width= 800,
        height = 800,
    )

    fig.show()
    #pio.write_html(fig, file=f"genres_proportions_{show_legend}.html", auto_open=False, auto_play=False)