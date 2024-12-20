import numpy as np
import pandas as pd

import plotly.graph_objects as go
from plotly.subplots import make_subplots

import json

from src.data.merge_genres import get_genres_merged
from src.utils.initial_plots_data import COLORS
from src.utils.save_plot import save_plot

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
    """
    Splits emotion scores into three parts (beginning, middle, end) for each row in the DataFrame.

    Args:
        df (DataFrame): DataFrame containing sentences and emotion scores.
        group_emotions (bool): Whether to group emotions into situations ("Positive", "Negative", "Plot Twist").

    Returns:
        new_df (DataFrame): A new DataFrame with added columns for each emotion's split scores.
    """
    
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
    """
    Adds new columns for grouped emotions ("Positive", "Negative", "Plot Twist") by summing relevant emotion scores.

    Args:
        df (DataFrame): DataFrame with emotion scores.

    Returns:
        separated_emotions_df (DataFrame): Modified DataFrame with additional columns for grouped emotions.
    """
    separated_emotions_df = df.copy()
    for key, value in map_situation_emotion.items():
        separated_emotions_df[str(key)] = separated_emotions_df[value].sum(axis=1)
    return separated_emotions_df

def split_movies_emotions_and_genres(plot_emotions_df, df_genres, group_emotions=False): 
    """
    Splits emotions into parts (beginning, middle, end) and merges with genres data.

    Args:
        plot_emotions_df (DataFrame): DataFrame containing emotion predictions.
        df_genres (DataFrame): DataFrame containing genre information.
        group_emotions (bool): Whether to group emotions into situations.

    Returns:
        DataFrame: A merged and exploded DataFrame containing genres and split emotions.
    """
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

def get_transition_matrix(df, from_, to, group_emotions=False):
    """
    Creates a transition matrix that shows probabilities of moving from one emotion to another 
    between two parts of a movie.

    Args:
        df (DataFrame): DataFrame containing split emotion data.
        from_ (int): The source part (e.g., beginning or middle).
        to (int): The target part (e.g., middle or end).
        group_emotions (bool): Whether to group emotions into situations.

    Returns:
        DataFrame: A transition matrix where rows are source emotions and columns are target emotions.
    """
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
    df_temp[f'principal_emotion_{from_}'] = df[[emotion+f'_{from_}' for emotion in emotions_list]].idxmax(axis=1)
    df_temp[f'principal_emotion_{to}'] = df[[emotion+f'_{to}' for emotion in emotions_list]].idxmax(axis=1)
    
    for idx, row in df_temp.iterrows():
        emotion_1 = row[f'principal_emotion_{from_}'][:-2]
        emotion_2 = row[f'principal_emotion_{to}'][:-2]
    
        i_1 = map_to_idx.get(emotion_1)
        i_2 = map_to_idx.get(emotion_2)
    
        transition_matrix[i_1, i_2] += 1
    
    for i in range(N):
        transition_matrix[i, :] /= sum(transition_matrix[i, :]) if sum(transition_matrix[i, :]) != 0 else 1

    return pd.DataFrame(transition_matrix)

def plot_heat_map_transitions_plotly(df, file_name="_heatmap_transitions", genre="All Genres", group_emotions=False):
    """
    Creates a heatmap visualization of transition probabilities between emotions or situations
    and saved it in an HTML file.

    Args:
        df (DataFrame): DataFrame containing emotion data.
        by_genre (bool): Whether to create separate heatmaps for each genre.
        file_name (str): Name of the HTML file to save the heatmap.
        group_emotions (bool): Whether to group emotions into situations (Positive, Negative, Surprise).

    """
    if group_emotions:
        map_to_idx = map_idx_situation
        map_to_emotion = idx_to_situation
    else:
        map_to_idx = map_idx_emotion
        map_to_emotion = idx_to_emotion

    if genre=="All Genres":
        df_genre = df
    else:
        df_genre = df.loc[df.genres==genre]

    transitions_df = get_transition_matrix(df_genre, 2, 3, group_emotions=group_emotions).add(get_transition_matrix(df_genre, 1, 2, group_emotions=group_emotions), fill_value=0).multiply(0.5)
    title = f"Most common transitions in {genre}"

    # Add a text representation of probabilities
    text = transitions_df.map(lambda x: f"{x:.2}") 

    figure = (title, transitions_df, text, genre)
    
    fig = go.Figure()
    fig.add_trace(
        go.Heatmap(
            z=transitions_df.values,
            x=[map_to_emotion[i] for i in range(len(map_to_emotion))],
            y=[map_to_emotion[i] for i in range(len(map_to_emotion))],
            colorscale='Greens',
            colorbar=dict(title='Transition Probability'),
            text=text.values,  # Add the text to display probabilities
            texttemplate="%{text}",  # Format to show the text directly on the heatmap
            textfont=dict(size=10),
            hovertemplate=(
                "<b>From</b>: %{y}<br>" 
                "<b>To</b>: %{x}<br>"   
                "<b>Probability</b>: %{z:.2f}<extra></extra>"  
            ),
        )
    )
    
    
    # Update layout with dropdown menu
    fig.update_layout(
        title=f"Emotion Transition Heatmaps for {genre} movies",
        xaxis=dict(title='To...'),
        yaxis=dict(title='From...', automargin=True),
        coloraxis_colorbar=dict(
            x=1.05  
        ),
        margin=dict(l=120, r=80, t=50, b=50) 
    )
    
    # Save the plot as an HTML file
    if group_emotions:
        save_plot(fig, genre+'_grouped_'+file_name)
    else:
        save_plot(fig, genre+file_name)

    # Force the width and the height for the notebook
    fig.update_layout(
        width= 700,
        height = 400,
    )
    fig.show()



def plot_separated_sankey_plotly(emotions_split_df, genre="All Genres"):
    """
    Creates a Sankey diagram to visualize transitions between emotions across movie parts
    and save it in an HTML file. Separate 1->2 and 2->3 and gives the choice of the genre 
    in a dropdown menu.

    Args:
        emotions_split_df (DataFrame): DataFrame containing split emotion data.
    """
    genres = emotions_split_df["genres"].unique()
    
    # Define states and colors
    states = emotion_columns  # Define a single column of states
    node_colors = [COLORS[state.split()[0]] for state in states]
    
    fig = go.Figure()
    fig = make_subplots(
        rows=1,
        cols=2,
        specs=[[{"type": "domain"}, {"type": "domain"}]], 
    )

    if genre=="All Genres":
        genre_df = emotions_split_df
    else:
        genre_df = emotions_split_df[emotions_split_df["genres"]==genre]
    
    # Transition matrices
    matrix1_to_2 = np.array(get_transition_matrix(genre_df, 1, 2))
    matrix2_to_3 = np.array(get_transition_matrix(genre_df, 2, 3))

    # Convert matrices to links
    def matrix_to_links(matrix, offset_source, offset_target):
        links = []
        for i, row in enumerate(matrix):
            total_outgoing = sum(row)
            for j, value in enumerate(row):
                if value > 0:
                    links.append({
                        "source": i + offset_source,
                        "target": j + offset_target,
                        "value": value
                    })
        return links

    links1_to_2 = matrix_to_links(matrix1_to_2, offset_source=0, offset_target=len(states))
    link_colors1 = [node_colors[link["source"]] for link in links1_to_2]

    links2_to_3 = matrix_to_links(matrix2_to_3, offset_source=0, offset_target=len(states))
    link_colors2 = [node_colors[link["source"]] for link in links2_to_3]

    fig.add_trace(
        go.Sankey(
            visible=False,
            arrangement="snap",
            node=dict(
                label=states + states,  
                color=node_colors * 2,  
                pad=20,
                thickness=20,
                hovertemplate="<b>Emotion:</b> %{label}<extra></extra>"
            ),
            link=dict(
                source=[link["source"] for link in links1_to_2],
                target=[link["target"] for link in links1_to_2],
                value=[link["value"] for link in links1_to_2],
                color=link_colors1,
                hovertemplate=(
                    "<b>From:</b> %{source.label}<br>"
                    "<b>To:</b> %{target.label}<br>"
                    "<b>Value:</b> %{value:.0%}<extra></extra>"  
                )
            ),
            domain=dict(
                x=[0.0, 0.5], 
                y=[0.0, 1.0]
            )
        ),
        row=1, col=1
    )

    fig.add_trace(
        go.Sankey(
            visible=False,
            arrangement="snap",
            node=dict(
                label=states + states,  
                color=node_colors * 2,  
                pad=20,
                thickness=20,
                hovertemplate="<b>Emotion:</b> %{label}<extra></extra>"
            ),
            link=dict(
                source=[link["source"] for link in links2_to_3],
                target=[link["target"] for link in links2_to_3],
                value=[link["value"] for link in links2_to_3],
                color=link_colors2,
                hovertemplate=(
                    "<b>From:</b> %{source.label}<br>"
                    "<b>To:</b> %{target.label}<br>"
                    "<b>Value:</b> %{value:.0%}<extra></extra>"  
                )
            ),
            domain=dict(
                x=[0.5, 1.0], 
                y=[0.0, 1.0]
            )
        ),
        row=1, col=2
    )
    
    # Make the first genre visible
    fig.data[0].visible = True
    fig.data[1].visible = True

    # Create dropdown menu
    buttons = [
        dict(
            label=genre,
            method="update",
            args=[
                {"visible": [
                    (i == idx * 2) or (i == idx * 2 + 1)  
                    for i in range(len(genres) * 2)
                ]},
                {"title": f"Sankey Diagram for Genre: {genre}"}
            ]
        )
        for idx, genre in enumerate(genres)
    ]

    fig.update_layout(
        title=dict(
            text=f"Transitions throughout {genre} movies",
            x=0.5,  # Center the title
            font=dict(size=16)
        ),
        annotations=[
            dict(
                x=0.25, y=1.1,  
                text="Transitions: Beginning to Middle",
                showarrow=False,
                font=dict(size=14),
                xanchor="center"
            ),
            dict(
                x=0.75, y=1.1,  
                text="Transitions: Middle to End",
                showarrow=False,
                font=dict(size=14),
                xanchor="center"
            ),
            dict(x=0.02, y=-0.1, text="Beginning", showarrow=False, font=dict(size=14), xanchor="center"),
            dict(x=0.43, y=-0.1, text="Middle", showarrow=False, font=dict(size=14), xanchor="center"),
            dict(x=0.57, y=-0.1, text="Middle", showarrow=False, font=dict(size=14), xanchor="center"),
            dict(x=0.98, y=-0.1, text="End", showarrow=False, font=dict(size=14), xanchor="center"),
        ],
        margin=dict(l=20, r=20, t=80, b=50) 
    )

    save_plot(fig, genre + "_sankey_charts_transitions")

    # Force the width and the height for the notebook
    fig.update_layout(
        width= 700,
        height = 400,
    )

    fig.show()