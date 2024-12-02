import pandas as pd
import numpy as np
import json
import plotly.graph_objects as go
from plotly.subplots import make_subplots

from src.data.load_final_data import load_final_movies_and_reviews
from src.data.merge_genres import get_genres_merged

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

# Expand the 'emotion_predictions' column into multiple columns
def separate_by_column(row):
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

def construct_emotions_by_genre_and_time_df(plot_emotions_df, df_genres):
    separated_emotions = plot_emotions_df['emotion_predictions'].apply(separate_by_column).apply(pd.Series)
    plot_emotions_df = pd.concat([plot_emotions_df.drop(columns=['emotion_predictions']), separated_emotions], axis=1)

    plot_emotions_df['n_sentences'] = plot_emotions_df.sentences.apply(lambda x: len(x)).copy()
    indices_to_drop = plot_emotions_df[(plot_emotions_df.n_sentences.eq(5) | plot_emotions_df.n_sentences.eq(8))].index
    plot_emotions_df.drop(indices_to_drop, inplace=True)

    plot_emotions_split_df = split_plot_emotions(plot_emotions_df, n=3)

    df_genres_grouped = get_genres_merged(df_genres)

    # get rid of movies that have no genre assigned to it
    df_genres_grouped = df_genres_grouped[(df_genres_grouped.drop(columns=['wikipedia_ID']).any(axis=1))]
    
    # create a column genres that contains a list of genre for each movie
    bool_cols = df_genres_grouped.select_dtypes(include=["bool"]).columns
    df_genres_grouped['genres'] = df_genres_grouped[bool_cols].copy().apply(lambda row: row.index[row].tolist(), axis=1)
    
    df_genres_emotions = df_genres_grouped[['wikipedia_ID', 'genres']].merge(plot_emotions_split_df, on='wikipedia_ID')

    # Step 1: Explode genres
    df_genres_emotions_exploded = df_genres_emotions.explode('genres')
    
    # Step 2: Melt DataFrame to structure emotions and time
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
    
    # Step 3: Calculate mean values for each genre, time, and emotion
    df_emotions_by_genre_time = melted.groupby(['genres', 'time', 'emotion'], as_index=False)['value'].mean()

    # Pivoting the dataframe to have emotions as columns
    df_emotions_by_genre_time = df_emotions_by_genre_time.pivot(index=['genres', 'time'], columns='emotion', values='value').reset_index()
    
    # Renaming the columns
    df_emotions_by_genre_time.columns.name = None
    df_emotions_by_genre_time = df_emotions_by_genre_time.rename_axis(None, axis=1)

    return df_emotions_by_genre_time

def plot_bar_and_scatter_emotion_evolution(df_emotions_by_genre_time):
    emotion_columns = ['anger', 'disgust', 'fear', 'joy', 'sadness', 'surprise']
    parts = {1: "Beginning", 2: "Middle", 3: "End"}
    genres_list = df_emotions_by_genre_time['genres'].unique()

    # Initialize the plots
    fig = make_subplots(
        rows=1, cols=2,
        subplot_titles=(None, None),
        specs=[[{"type": "bar"}, {"type": "scatter"}]]
    )
    fig.add_trace(
        go.Bar(
        ),
        row=1, col=1
    )
    for emotion in emotion_columns:
        fig.add_trace(
            go.Scatter(
            ),
            row=1, col=2
        )
    
    frames = []
    dropdown_buttons = []
    periods = [1, 2, 3]
    
    for genre in genres_list:
        genre_slider_steps = []
        df_used = df_emotions_by_genre_time[df_emotions_by_genre_time['genres'] == genre].drop(columns='genres')
    
        periodic_emotions = df_used.groupby("time")[emotion_columns].mean()
    
        periodic_emotions_diff = periodic_emotions.pct_change().fillna(0) * 100
        periodic_emotions_diff.iloc[0] = 0
    
        for period in periods:
            if period in periodic_emotions.index:
                emotion_means = periodic_emotions.loc[period]
                emotion_variations = periodic_emotions_diff.loc[period]
                
                dominant_emotion = emotion_variations.idxmax()
                dominant_color = emotion_colors[dominant_emotion]
                dominant_emotion_name = dominant_emotion
        
                n_emotions = len(emotion_columns)
                coords = {emotion_means.iloc[i]: emotion_columns[i] for i in range(n_emotions)}
                emotion_means = list(emotion_means)
                emotion_means.sort()
                emotion_columns_sorted = [coords.get(emo) for emo in emotion_means]
        
                barplot_data = go.Bar(
                    x=emotion_columns_sorted,
                    y=emotion_means,
                    marker=dict(color=[emotion_colors[col] for col in emotion_columns_sorted]),
                    name=f"Mean score ({period})"
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
                            size=abs(value) * 0.5 + 10,  # Ajustement de la taille des marqueurs
                            color=emotion_colors[emotion],
                            symbol="triangle-up" if value > 0 else "triangle-down",
                        ),
                        name=f"Variation ({period})"
                    ))
                
                frames.append(go.Frame(
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
                                text=f"<b>{parts[period]} of the movie<br>Most significant variation from previous part : {dominant_emotion_name} {emotion_emojis[dominant_emotion_name]}</b>",
                                font=dict(size=18, color=dominant_color, family="Arial, sans-serif")
                            )
                        ]
                    )
                ))
        
                genre_slider_steps.append(
                    dict(
                        args=[
                            [f"{genre}_{period}"],
                            {"frame": {"duration": 500, "redraw": True}, "mode": "immediate"}
                        ],
                        label=parts.get(period),
                        method="animate"
                    )
                )
    
        dropdown_buttons.append(
            dict(
                label=genre,
                method="update",
                args=[
                    {"frames": [frame.name for frame in frames]},  # Activate only the genre frames
                    {"sliders": [dict(steps=genre_slider_steps)]}  # Activate only the genre steps
                ]
            )
        )
    
    fig.update(frames=frames)
    
    fig.update_layout(
        updatemenus=[
            dict(
                type="dropdown",
                showactive=True,
                buttons=dropdown_buttons,
                x=0.5,
                xanchor="center",
                y=1.3,
                yanchor="top"
            )
        ]
    )
    
    fig.update_layout(
        width=1200,
        height=600,
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
            steps=[
                dict(
                    args=[[period], {"frame": {"duration": 500, "redraw": True}, "mode": "immediate"}],
                    label=parts[period],  
                    method="animate"
                )
                for period in periods
            ]
        )
    ]
    )
    
    fig.add_annotation(
        x=0.5,  
        y=-0.6, 
        text="<b>Distribution and Variation of emotion thoughout the movies.</b>",
        showarrow=False,
        xref="paper",
        yref="paper",
        font=dict(size=25, color='black', family="Arial"),
        align="center"
    )
    
    fig.update_yaxes(
        range=[-70, 70], 
        title="Variation (%)",
        row=1,
        col=2,
        showticklabels=False,
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
        
    # Afficher le graphique
    fig.show()