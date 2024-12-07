import plotly.graph_objs as go
from plotly.subplots import make_subplots
import plotly.graph_objects as go

from src.utils.plot_3D_actor_emotions import *
from src.utils.initial_plots_data import COLORS, EMOJIS
from src.utils.save_plot import save_plot

TOP_ACTORS_MOVIES_COUNT = 100

def plot_actors_emotion_selector(df_characters, df_movies_with_emotions, df_reviews_with_emotions):
    """
    Given the the actors and their associated movies, along the movies plot and reviews emotions.
    Plot two bar plot displaying the mean percentage emotions for the top TOP_ACTORS_MOVIES_COUNT
    actors that played in the most movies in our dataset (can be selected through a dropdown menu)

    Arguments:
        - df_characters: Dataframe containing the actors and character information associated with the movie they played in
        - df_movies_with_emotions: Dataframe containing for each movies the associated emotions of its plot
        - df_reviews_with_emotions: Dataframe containing the emotions associated to each reviews of a movies

    Returns two dataframes containing for each emotion the actor that has the maximum mean emotion from the top 
        TOP_ACTORS_MOVIES_COUNT actors that played in the most movies in our dataset.
        - df_emotion_max_actor_plots: Dataframe containing the emotions for the plots
        - df_emotion_max_actor_reviews: Dataframe containing the emotions for the reviews.
    """
    df_actor_mean_emotions_plot, df_actor_emotions_plot = get_mean_plot_actor_emotion(df_characters, df_movies_with_emotions)
    df_actor_mean_emotions_reviews, df_actor_emotions_reviews = get_mean_review_actor_emotion(df_characters, df_reviews_with_emotions)

    # Remove 'mean_' before column names
    df_actor_mean_emotions_plot.columns = df_actor_mean_emotions_plot.columns.str.replace("mean_", "").str.capitalize()
    df_actor_mean_emotions_reviews.columns = df_actor_mean_emotions_reviews.columns.str.replace("mean_", "").str.capitalize()

    # To percentage
    df_actor_mean_emotions_plot = (df_actor_mean_emotions_plot * 100).round(2)
    df_actor_mean_emotions_reviews = (df_actor_mean_emotions_reviews * 100).round(2)

    # Limit the number of actors to the top TOP_ACTORS_MOVIES_COUNT actors that played in the most movies
    actors_ID = pd.Series(list(set(df_actor_mean_emotions_reviews.index).intersection(set(df_actor_mean_emotions_plot.index))))

    count_actor_movies = df_actor_emotions_plot.groupby("freebase_ID_actor")["wikipedia_ID"].count()
    count_actor_movies = count_actor_movies[actors_ID]

    top_actors = count_actor_movies.sort_values(ascending=False)[:TOP_ACTORS_MOVIES_COUNT]

    df_id_actor_name = df_characters[["actor_name", "freebase_ID_actor"]].drop_duplicates(subset=["freebase_ID_actor"]).set_index("freebase_ID_actor")
    df_actors_to_display = df_id_actor_name.loc[top_actors.index]

    # Display bars
    first_to_display = df_actors_to_display.iloc[0].name

    # https://plotly.com/python/subplots/
    fig = make_subplots(rows=1, cols=2, subplot_titles=["Actor's mean movie plot emotion percentage", "Actor's mean movie reviews emotion percentage"])
    fig.add_trace(
        go.Bar(x=df_actor_mean_emotions_plot.columns, y=df_actor_mean_emotions_plot.loc[first_to_display]), row=1, col=1
    )
    fig.add_trace(
        go.Bar(x=df_actor_mean_emotions_reviews.columns, y=df_actor_mean_emotions_reviews.loc[first_to_display]), row=1, col=2
    )

    # https://plotly.com/python/dropdowns/
    # https://stackoverflow.com/questions/68685246/how-to-modify-a-plotly-graphs-layout-using-custom-buttons
    # Create drop down
    max_pct = 0
    buttons = []
    for id_actor, actor_name in df_actors_to_display.iterrows():
        actor_name = actor_name.iloc[0]
        y_plot_emotions = df_actor_mean_emotions_plot.loc[id_actor]
        y_reviews_emotions = df_actor_mean_emotions_reviews.loc[id_actor]
        max_pct = max(max(y_plot_emotions), max(y_reviews_emotions), max_pct)

        top_plot_emotion_emoji = EMOJIS[y_plot_emotions.idxmax().lower()]
        top_review_emotion_emoji = EMOJIS[y_reviews_emotions.idxmax().lower()]
        
        label_button = f"{actor_name} {top_plot_emotion_emoji} {top_review_emotion_emoji}"
        button = dict(label=label_button, method='update', args=[{'y': [df_actor_mean_emotions_plot.loc[id_actor], df_actor_mean_emotions_reviews.loc[id_actor]], 'title': actor_name}])
        buttons.append(button)

    fig.update_layout(
        updatemenus=[
            dict(
                buttons=buttons,
                direction="down",
                showactive=True,
                x=0.5,
                xanchor="center",
                y=1.15,
                yanchor="middle"
            ),
        ]
    )
    range_y_axis = [0,max_pct*1.10]
    fig.update_yaxes(range=range_y_axis, title="Mean emotion percentage", row=1, col=1, gridcolor='lightgrey')
    fig.update_yaxes(range=range_y_axis, row=1, col=2, gridcolor='lightgrey')
    fig.update_layout(
        plot_bgcolor='white'
    )

    # Title
    fig.update_layout(title="Actor mean movie's plot and reviews emotion", title_x=0.5, title_y=0.95)

    # Legend
    fig.update_layout(legend_visible=False)

    fig.update_traces(hovertemplate="%{x}<br>%{y}%<extra></extra>")

    # Set colors
    # https://community.plotly.com/t/plotly-express-bar-colour-change/36210/7
    colors = [COLORS[color.lower()] for color in df_actor_mean_emotions_plot.columns]
    fig.update_traces(marker_color=colors)

    # Set ticks labels
    # https://stackoverflow.com/questions/72463669/how-to-set-xaxis-ticks-in-plotly
    ticks = [10, 20, 30, 40, 50]
    fig.update_yaxes(tickmode='array', tickvals = ticks, ticktext = [f"{x}% " for x in ticks])

    fig.show()

    # Save plot
    save_plot(fig, figure_name="actor_selector_emotions_profile")

    # Output for each emotion the actor that maximize it for both the reviews and the plots
    df_emotion_max_actor_plots = df_actors_to_display.merge(df_actor_mean_emotions_plot, left_index=True, right_index=True)
    df_emotion_max_actor_plots = df_emotion_max_actor_plots.set_index("actor_name")
    df_emotion_max_actor_plots_max = pd.concat([df_emotion_max_actor_plots.idxmax(), df_emotion_max_actor_plots.max()], axis=1)
    df_emotion_max_actor_plots_max.columns = ["emotion", "emotion_value"]

    df_emotion_max_actor_reviews = df_actors_to_display.merge(df_actor_mean_emotions_reviews, left_index=True, right_index=True)
    df_emotion_max_actor_reviews = df_emotion_max_actor_reviews.set_index("actor_name")
    df_emotion_max_actor_reviews_max = pd.concat([df_emotion_max_actor_reviews.idxmax(), df_emotion_max_actor_reviews.max()], axis=1)
    df_emotion_max_actor_reviews_max.columns = ["emotion", "emotion_value"]

    return df_emotion_max_actor_plots_max, df_emotion_max_actor_reviews_max