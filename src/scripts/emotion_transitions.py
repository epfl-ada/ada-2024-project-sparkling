import numpy as np
import pandas as pd

import plotly.graph_objects as go
from plotly.subplots import make_subplots
import plotly.io as pio

import json

from src.data.merge_genres import get_genres_merged

emotion_columns = ['anger', 'disgust', 'fear', 'joy', 'sadness', 'surprise']
situation_columns = ['Positive', 'Negative', 'Plot Twist']

map_idx_emotion = {
    'anger': 0,
    'disgust': 1,
    'fear': 2,
    'joy': 3,
    'sadness': 4,
    'surprise': 5
}

map_idx_situation = {
    'Positive': 0,
    'Negative': 1,
    'Plot Twist': 2
}

map_situation_emotion = {
    'Positive': ['joy'],
    'Negative': ['anger', 'fear', 'disgust', 'sadness'],
    'Plot Twist': ['surprise']
}

idx_to_emotion = {v: k for k, v in map_idx_emotion.items()}
idx_to_situation = {v: k for k, v in map_idx_situation.items()}

def separate_by_column(row):
    data = json.loads(row)
    return {
        "sentences": data.get("sentences", []),
        "anger": data.get("anger", []),
        "disgust": data.get("disgust", []),
        "fear": data.get("fear", []),
        "joy": data.get("joy", []),
        "sadness": data.get("sadness", []),
        "surprise": data.get("surprise", [])
    }

def split_plot_emotions(df, group_emotions=False):
    if group_emotions:
        emotions_list = situation_columns
    else:
        emotions_list = emotion_columns
        
    new_df = df.copy()

    for i in range(3):

        new_columns = [emotion+'_'+str(i+1) for emotion in emotions_list]
        new_df[new_columns] = 0.0
    
        for idx, row in new_df.iterrows():
            total = len(row.sentences)
            splits = [0, int(total * 0.25), int(total * 0.75), total]

            for emotion in emotions_list:
                new_df.loc[idx, emotion+'_'+str(i+1)] = np.mean(row[emotion][splits[i]:splits[i+1]])
                
    return new_df

def add_pos_neg_plot_twist(df):
    separated_emotions_df = df.copy()
    for key, value in map_situation_emotion.items():
        separated_emotions_df[str(key)] = separated_emotions_df[value].sum(axis=1)
    return separated_emotions_df

def split_movies_emotions_and_genres(plot_emotions_df, df_genres, group_emotions=False): 
    separated_emotions = plot_emotions_df['emotion_predictions'].apply(separate_by_column).apply(pd.Series)
    
    if group_emotions:
        separated_emotions = add_pos_neg_plot_twist(separated_emotions)
    
    plot_emotions_df = pd.concat([plot_emotions_df.drop(columns=['emotion_predictions']), separated_emotions], axis=1)

    plot_emotions_df['n_sentences'] = plot_emotions_df.sentences.apply(lambda x: len(x)).copy()
    indices_to_drop = plot_emotions_df[(plot_emotions_df.n_sentences.eq(5) | plot_emotions_df.n_sentences.eq(8))].index
    plot_emotions_df.drop(indices_to_drop, inplace=True)

    plot_emotions_split_df = split_plot_emotions(plot_emotions_df, group_emotions=group_emotions)

    df_genres_grouped = get_genres_merged(df_genres)

    # get rid of movies that have no genre assigned to it
    df_genres_grouped = df_genres_grouped[(df_genres_grouped.drop(columns=['wikipedia_ID']).any(axis=1))]
    
    # create a column genres that contains a list of genre for each movie
    bool_cols = df_genres_grouped.select_dtypes(include=["bool"]).columns
    df_genres_grouped['genres'] = df_genres_grouped[bool_cols].copy().apply(lambda row: row.index[row].tolist(), axis=1)
    
    df_genres_emotions = df_genres_grouped[['wikipedia_ID', 'genres']].merge(plot_emotions_split_df, on='wikipedia_ID')

    return df_genres_emotions.explode("genres")

def get_transition_matrix(df, group_emotions=False):
    if group_emotions:
        emotions_list = situation_columns
        map_to_idx = map_idx_situation
        map_to_emotion = idx_to_situation
    else:
        emotions_list = emotion_columns
        map_to_idx = map_idx_emotion
        map_to_emotion = idx_to_emotion

    N = len(emotions_list)
    transition_matrix = np.zeros((N, N))
    
    df_temp = df.copy()
    for i in [1, 2, 3]:
        df_temp[f'principal_emotion_{i}'] = df[[emotion+f'_{i}' for emotion in emotions_list]].idxmax(axis=1)
    for idx, row in df_temp.iterrows():
        emotion_1 = row.principal_emotion_1[:-2]
        emotion_2 = row.principal_emotion_2[:-2]
        emotion_3 = row.principal_emotion_3[:-2]
    
        i_1 = map_to_idx.get(emotion_1)
        i_2 = map_to_idx.get(emotion_2)
        i_3 = map_to_idx.get(emotion_3)
    
        transition_matrix[i_1, i_2] += 1
        transition_matrix[i_2, i_3] += 1
    
    for i in range(N):
        transition_matrix[i, :] /= sum(transition_matrix[i, :]) if sum(transition_matrix[i, :]) != 0 else 1

    return pd.DataFrame(transition_matrix)

def plot_heat_map_transitions_plotly(transitions_df, title, group_emotions=False):
    if group_emotions:
        map_to_idx = map_idx_situation
        map_to_emotion = idx_to_situation
    else:
        map_to_idx = map_idx_emotion
        map_to_emotion = idx_to_emotion
        
    fig = go.Figure(
        data=go.Heatmap(
            z=transitions_df.values,
            x=[map_to_emotion[i] for i in range(len(map_))],
            y=[map_to_emotion[i] for i in range(len(map_))],
            colorscale='plasma',
            colorbar=dict(title='Transition Probability'),
        )
    )
    fig.update_layout(
        title=title,
        xaxis=dict(title='To...'),
        yaxis=dict(title='From...', automargin=True),
    )
    return fig


def heat_map_transitions_plotly(df, by_genre=False, file_name="heatmap_transitions.html", group_emotions=False):
    if group_emotions:
        map_to_idx = map_idx_situation
        map_to_emotion = idx_to_situation
    else:
        map_to_idx = map_idx_emotion
        map_to_emotion = idx_to_emotion
        
    figures = []
    genres_list = ['All Movies'] + (list(df.genres.unique()) if by_genre else [])
    
    # Create data for each genre
    for genre in genres_list:
        if genre == 'All Movies':
            transitions_df = get_transition_matrix(df, group_emotions=group_emotions)
            title = "Most common transitions among all movies."
        else:
            df_genre = df.loc[df.genres == genre]
            transitions_df = get_transition_matrix(df_genre, group_emotions=group_emotions)
            title = f"Most common transitions in {genre}"
        figures.append((title, transitions_df, genre))
    
    # Create dropdown menu
    fig = go.Figure()

    # Add traces for each genre
    for idx, (title, transitions_df, genre) in enumerate(figures):
        fig.add_trace(
            go.Heatmap(
                z=transitions_df.values,
                x=[map_to_emotion[i] for i in range(len(map_to_emotion))],
                y=[map_to_emotion[i] for i in range(len(map_to_emotion))],
                colorscale='plasma',
                visible=idx == 0,  # Only show the first heatmap initially
                colorbar=dict(title='Transition Probability') if idx == 0 else None,
            )
        )
    
    # Update layout with dropdown menu
    fig.update_layout(
        title="Emotion Transition Heatmaps",
        xaxis=dict(title='To...'),
        yaxis=dict(title='From...', automargin=True),
        updatemenus=[
            dict(
                buttons=[
                    dict(
                        label=genre,  # Use genre name as the label
                        method="update",
                        args=[
                            {"visible": [i == idx for i in range(len(figures))]},  # Show only the selected genre
                            {"title": title},
                        ],
                    )
                    for idx, (title, _, genre) in enumerate(figures)
                ],
                direction="down",
                showactive=True,
            )
        ],
    )
    
    # Save the plot as an HTML file
    if group_emotions:
        pio.write_html(fig, file='grouped_'+file_name, auto_open=False)
        print(f"Heatmap saved as grouped_{file_name}")
    else:
        pio.write_html(fig, file=file_name, auto_open=False)
        print(f"Heatmap saved as {file_name}")