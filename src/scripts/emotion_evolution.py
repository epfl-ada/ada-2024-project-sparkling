import pandas as pd
import numpy as np
import json
import plotly.graph_objects as go
from plotly.subplots import make_subplots


from src.data.load_final_data import load_final_movies_and_reviews
from src.data.merge_genres import get_genres_merged
from src.utils.save_plot import save_plot

emotion_colors = {
    "joy": "#f3c35f",
    "anger": "#e13c29",
    "disgust": "#8bc664",
    "sadness": "#85b8ed",
    "fear": "#b99be5",
    "surprise": "#de9f81"
}

emotion_emojis = {
    "anger": "😠",
    "disgust": "🤢",
    "fear": "😨",
    "joy": "😊",
    "sadness": "😢",
    "surprise": "😲"
}

emotion_columns = ['anger', 'disgust', 'fear', 'joy', 'sadness', 'surprise']

periods = [1, 2, 3]
parts = {1: "Beginning", 2: "Middle", 3: "End"}

def separate_by_column(row):
    """
    This function processes a JSON-formatted string (row), parses it into a Python dictionary,
    and extracts specific fields related to emotion analysis, organizing them into a new dictionary.

    Args:
        row (str): A JSON-formatted string containing data about sentences and their associated emotions.

    Returns:
        dict: A dictionary with keys for sentences and  emotions ("anger", "disgust", "fear", 
              "joy", "sadness", "surprise"). Each key maps to a list extracted from the JSON input.
    """
    data = json.loads(row)
    sentences = data.get("sentences", [])
    emotions = data.get("emotions", {})
    return {
        "sentences": data.get("sentences", []),
        "anger": data.get("anger", []),
        "disgust": data.get("disgust", []),
        "fear": data.get("fear", []),
        "joy": data.get("joy", []),
        "neutral": data.get("neutral", []),
        "sadness": data.get("sadness", []),
        "surprise": data.get("surprise", [])
    }

def split_plot_emotions(df, n=3):
    """
    Splits emotion scores into three parts (beginning, middle, end) for each row in the DataFrame.

    Args:
        df (DataFrame): DataFrame containing sentences and emotion scores.
        group_emotions (bool): Whether to group emotions into situations ("Positive", "Negative", "Plot Twist").

    Returns:
        new_df (DataFrame): A new DataFrame with added columns for each emotion's split scores.
    """
    new_df = df.copy()

    for i in range(n):
        new_df[['anger_'+str(i+1), 'disgust_'+str(i+1), 'fear_'+str(i+1), \
                          'joy_'+str(i+1), 'neutral_'+str(i+1), 'sadness_'+str(i+1), 'surprise_'+str(i+1)]] = 0.0
    
        for idx, row in new_df.iterrows():
            total = row.n_sentences
            splits = [0, int(total * 0.25), int(total * 0.75), total]
            
            new_df.loc[idx, 'anger_'+str(i+1)] = np.mean(row.anger[splits[i]:splits[i+1]])
            new_df.loc[idx, 'disgust_'+str(i+1)] = np.mean(row.disgust[splits[i]:splits[i+1]])
            new_df.loc[idx, 'fear_'+str(i+1)] = np.mean(row.fear[splits[i]:splits[i+1]])
            new_df.loc[idx, 'joy_'+str(i+1)] = np.mean(row.joy[splits[i]:splits[i+1]]) 
            new_df.loc[idx, 'neutral_'+str(i+1)] = np.mean(row.neutral[splits[i]:splits[i+1]])
            new_df.loc[idx, 'sadness_'+str(i+1)] = np.mean(row.sadness[splits[i]:splits[i+1]])
            new_df.loc[idx, 'surprise_'+str(i+1)] = np.mean(row.surprise[splits[i]:splits[i+1]]) 

    return new_df

def display_text_variation(period, dominant_emotion_name):
    """
    Generates a descriptive text annotation for a specific period of the movie,
    indicating the most significant emotion variation.

    Args:
        period (int): The current part of the movie.
        dominant_emotion_name (str): Name of the emotion with the most significant variation.

    Returns:
        str: A formatted string describing the dominant emotion and its variation.
    """
    if period != 1 :
        return f"<b>{parts[period]} of the movie<br>Most significant variation from previous part : {dominant_emotion_name} {emotion_emojis[dominant_emotion_name]}</b>"
    else:
        return "<b>Beginning of the movie<br>No previous part, no variation from previous part :)</b>"

def return_dominant_color(period, dominant_emotion_name):
    """
    Determines the color associated with the dominant emotion for a given period.
    Uses black as a default for the first period, as there are no variations.

    Args:
        period (int): The current part of the movie.
        dominant_emotion_name (str): Name of the dominant emotion for the period.

    Returns:
        str: The color corresponding to the dominant emotion, or black for the first period.
    """
    dominant_color = emotion_colors[dominant_emotion_name]
    if period == 1:
        return "black"
    else:
        return dominant_color

def construct_emotions_by_genre_and_time_df(plot_emotions_df, df_genres):
    """
    Processes emotion data, splits it into parts (beginning, middle, end), merges it with genre data,
    and calculates mean score values for each genre and time period.

    Args:
        plot_emotions_df (DataFrame): DataFrame containing raw emotion predictions.
        df_genres (DataFrame): DataFrame containing genre information.

    Returns:
        df_emotions_by_genre_time (DataFrame): A DataFrame containing average emotion scores grouped by genre and time period.
    """
    separated_emotions = plot_emotions_df['emotion_predictions'].apply(separate_by_column).apply(pd.Series)
    plot_emotions_df = pd.concat([plot_emotions_df.drop(columns=['emotion_predictions']), separated_emotions], axis=1)

    plot_emotions_df['n_sentences'] = plot_emotions_df.sentences.apply(lambda x: len(x)).copy()
    indices_to_drop = plot_emotions_df[(plot_emotions_df.n_sentences.eq(5) | plot_emotions_df.n_sentences.eq(8))].index
    plot_emotions_df.drop(indices_to_drop, inplace=True)

    plot_emotions_split_df = split_plot_emotions(plot_emotions_df, n=3)

    df_genres_grouped = get_genres_merged(df_genres)

    # Get rid of movies that have no genre assigned to it
    df_genres_grouped = df_genres_grouped[(df_genres_grouped.drop(columns=['wikipedia_ID']).any(axis=1))]
    
    # Create a column genres that contains a list of genre for each movie
    bool_cols = df_genres_grouped.select_dtypes(include=["bool"]).columns
    df_genres_grouped['genres'] = df_genres_grouped[bool_cols].copy().apply(lambda row: row.index[row].tolist(), axis=1)
    
    df_genres_emotions = df_genres_grouped[['wikipedia_ID', 'genres']].merge(plot_emotions_split_df, on='wikipedia_ID')

    # Explode genres
    df_genres_emotions_exploded = df_genres_emotions.explode('genres')
    
    # Melt DataFrame to structure emotions and time
    melted = df_genres_emotions_exploded.melt(
        id_vars=['genres', 'wikipedia_ID'],
        value_vars=[
            'anger_1', 'disgust_1', 'fear_1', 'joy_1', 'sadness_1', 'surprise_1',
            'anger_2', 'disgust_2', 'fear_2', 'joy_2', 'sadness_2', 'surprise_2',
            'anger_3', 'disgust_3', 'fear_3', 'joy_3', 'sadness_3', 'surprise_3'
        ],
        var_name='emotion_time',
        value_name='value'
    )
    
    # Extract emotion and time from `emotion_time`
    melted['emotion'] = melted['emotion_time'].str.extract(r'^(.*)_')[0]
    melted['time'] = melted['emotion_time'].str.extract(r'_(\d)$').astype(int)
    
    # Calculate mean values for each genre, time, and emotion
    df_emotions_by_genre_time = melted.groupby(['genres', 'time', 'emotion'], as_index=False)['value'].mean()

    # Pivoting the dataframe to have emotions as columns
    df_emotions_by_genre_time = df_emotions_by_genre_time.pivot(index=['genres', 'time'], columns='emotion', values='value').reset_index()
    
    # Renaming the columns
    df_emotions_by_genre_time.columns.name = None
    df_emotions_by_genre_time = df_emotions_by_genre_time.rename_axis(None, axis=1)

    return df_emotions_by_genre_time

def plot_bar_and_scatter_emotion_evolution(df_emotions_by_genre_time, genre, all_genres=False):
    """
    Creates a bar and scatter plot to visualize the evolution of emotions over time for a specific movie genre.
    Generates an interactive Plotly figure with animation frames and saves it as an HTML file.

    Args:
        df_emotions_by_genre_time (pd.DataFrame): DataFrame containing emotion data for different genres over time.
        genre (str): The genre to filter and plot. Ignored if `all_genres` is True.
        all_genres (bool): If True, includes all genres in the plot; otherwise, filters for the specified genre.
    """
    genre_filename_name = genre if genre is not None else "all_genres"
    file_name = f"{genre_filename_name}_bar_scatter_plot.html"
    genres_list = df_emotions_by_genre_time['genres'].unique()

    if all_genres:
        df_used = df_emotions_by_genre_time.drop(columns='genres')
    else:
        df_used = df_emotions_by_genre_time[df_emotions_by_genre_time['genres'] == genre].drop(columns='genres')

    periodic_emotions = df_used.groupby("time")[emotion_columns].mean()
    
    periodic_emotions_diff = periodic_emotions.pct_change().fillna(0) * 100
    periodic_emotions_diff.iloc[0] = 0

    # Initialize the plots
    fig = make_subplots(
        rows=1, cols=2,
        subplot_titles=(None, None),
        specs=[[{"type": "bar"}, {"type": "scatter"}]]
    )

    emotion_means = periodic_emotions.loc[1]
    emotion_variations = periodic_emotions_diff.loc[1]
                
    dominant_emotion = emotion_variations.idxmax()
        
    n_emotions = len(emotion_columns)
    coords = {emotion_means.iloc[i]: emotion_columns[i] for i in range(n_emotions)}
    emotion_means = list(emotion_means)
    emotion_means.sort()
    emotion_columns_sorted = [coords.get(emo) for emo in emotion_means]

    fig.add_trace(go.Bar(
        x=emotion_columns_sorted,
        y=emotion_means,
        marker=dict(color=[emotion_colors[col] for col in emotion_columns_sorted]),
        hovertemplate="Proportion : %{y:.2f}<extra></extra>",
        name="",
    ), row=1, col=1)
                
    scatter_data = []
    for emotion, value in emotion_variations.items():
        fig.add_trace(go.Scatter(
            x=[emotion],
            y=[value],
            text=[f"<b><span style='color: {'green' if value > 0 else 'red'};'>{value:+.2f}%</span> {emotion_emojis[emotion]}</b>"],
            mode="markers+text",
            textposition="bottom center" if value < 0 else "top center",
            marker=dict(
                size=abs(value) * 0.5 + 10, 
                color=emotion_colors[emotion],
                symbol="triangle-up" if value > 0 else "triangle-down",
            ),
                hovertemplate="Variation : %{y:.2f}%<extra></extra>",
            name="",
        ), row=1, col=2)

    frames = []
    slider_steps = []
    
    for period in periods:
        if period in periodic_emotions.index:
            emotion_means = periodic_emotions.loc[period]
            emotion_variations = periodic_emotions_diff.loc[period]
                
            dominant_emotion = emotion_variations.idxmax()
        
            n_emotions = len(emotion_columns)
            coords = {emotion_means.iloc[i]: emotion_columns[i] for i in range(n_emotions)}
            emotion_means = list(emotion_means)
            emotion_means.sort()
            emotion_columns_sorted = [coords.get(emo) for emo in emotion_means]
        
            barplot_data = go.Bar(
                x=emotion_columns_sorted,
                y=emotion_means,
                marker=dict(color=[emotion_colors[col] for col in emotion_columns_sorted]),
                hovertemplate="Proportion : %{y:.2f}<extra></extra>",
                name="", 
            )
                
            scatter_data = []
            for emotion, value in emotion_variations.items():
                scatter_data.append(go.Scatter(
                    x=[emotion],
                    y=[value],
                    text=[f"<b><span style='color: {'green' if value > 0 else 'red'};'>{value:+.2f}%</span> {emotion_emojis[emotion]}</b>"],
                    mode="markers+text",
                    textposition="bottom center" if value < 0 else "top center",
                    marker=dict(
                        size=abs(value) * 0.5 + 10, 
                        color=emotion_colors[emotion],
                        symbol="triangle-up" if value > 0 else "triangle-down",
                    ),
                    hovertemplate="Variation : %{y:.2f}%<extra></extra>",
                    name="" 
                ))

            frame = go.Frame(
                data=[barplot_data] + scatter_data,
                name=f"{genre}_{period}",
                layout=go.Layout(
                    annotations=[
                        dict(
                            x=0.5,
                            y=1.2,
                            showarrow=False,
                            xref="paper",
                            yref="paper",
                            text=display_text_variation(period, dominant_emotion),
                            font=dict(size=18, color=return_dominant_color(period, dominant_emotion), family="Arial, sans-serif")
                        )
                    ]
                )
            )

            frames.append(frame)

            slider_steps.append(
                dict(
                    args=[
                        [f"{genre}_{period}"],
                        {"frame": {"duration": 500, "redraw": True}, "mode": "immediate"}
                    ],
                    label=parts.get(period),
                    method="animate"
                )
            )

    fig.update(frames=frames)
    
    fig.update_layout(
        xaxis_title="Emotions",
        yaxis_title="Emotions mean proportion",
        xaxis2_title="Emotions",
        yaxis2_title="Variation (%)",
        template="plotly_white",
        showlegend=False,
        sliders=[
            dict(
                active=0,
                currentvalue={
                    "prefix": "Part: ",
                    "font": {"size": 16, "color": "black"}
                },
                pad={"t": 50},
                steps=slider_steps
            )
        ]
    )

    
    fig.add_annotation(
        x=0.5,  
        y=-0.6, 
        text="", #<b>Distribution and Variation of emotion throughout the movies.</b>",
        showarrow=False,
        xref="paper",
        yref="paper",
        font=dict(size=25, color='black', family="Arial"),
        align="center"
    )
    
    fig.update_yaxes(
        range=[-100, 100], 
        title="Variation (%)",
        row=1,
        col=2,
    )
    
    fig.update_yaxes(
        range=[0, 0.26], 
        row=1,
        col=1
    )
    fig.update_xaxes(
        range=[-0.5, len(emotion_columns) - 0.5],
        categoryorder='total ascending', 
        row=1,
        col=2,
        title=None
    )
    
    fig.update_xaxes(
        row=1,
        col=1,
        title=None
    )
    save_plot(fig, file_name)

    fig.show()
    
