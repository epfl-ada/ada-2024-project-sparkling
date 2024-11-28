from src.utils.initial_plots_data import COLORS
from src.data.normalize_emotions import normalize_total_plot_emotions, normalize_review_emotions
from sklearn.decomposition import PCA
import pandas as pd
import plotly.express as px

MIN_MOVIE_ACTOR = 3
MIN_REVIEWS_ACTOR = 10

def plot_3D_actor_plot_emotion(df_characters, df_movies_with_emotions):
    """
    Given the dataframe containing the actors name along with the movie they played in and 
    the emotions of the plot of the movies.
    Plot a 3D scatter of the actors according to their most present emotion in the movie plot they played in.
    Displays the 3D scatter plot by using PCA on the emotions means.
    """
    df_movies_with_emotions = normalize_total_plot_emotions(df_movies_with_emotions)[["wikipedia_ID", 
                                                                                    "normalized_plot_anger_without_neutral", 
                                                                                    "normalized_plot_disgust_without_neutral", 
                                                                                    "normalized_plot_fear_without_neutral", 
                                                                                    "normalized_plot_joy_without_neutral", 
                                                                                    "normalized_plot_sadness_without_neutral", 
                                                                                    "normalized_plot_surprise_without_neutral"]]
    df_actor_emotions = df_characters.merge(df_movies_with_emotions, on="wikipedia_ID")

    actors_min_movies = df_actor_emotions.groupby("freebase_ID_actor")["wikipedia_ID"].count() >= MIN_MOVIE_ACTOR
    df_actor_emotions = df_actor_emotions[df_actor_emotions["freebase_ID_actor"].isin(actors_min_movies[actors_min_movies].index)]

    df_actor_mean_emotions = df_actor_emotions.groupby("freebase_ID_actor").agg(mean_anger=("normalized_plot_anger_without_neutral", "mean"), 
                                                    mean_disgust=("normalized_plot_disgust_without_neutral", "mean"),
                                                    mean_fear=("normalized_plot_fear_without_neutral", "mean"),
                                                    mean_joy=("normalized_plot_joy_without_neutral", "mean"),
                                                    mean_sadness=("normalized_plot_sadness_without_neutral", "mean"),
                                                    mean_surprise=("normalized_plot_surprise_without_neutral", "mean"))
    
    colors_label = df_actor_mean_emotions.idxmax(axis=1).str.replace("mean_","")

    pca = PCA(n_components=3)
    pd_pca_actor_means_emotion = pd.DataFrame(pca.fit_transform(df_actor_mean_emotions), index=df_actor_mean_emotions.index).reset_index()

    # Merge data we want to display
    df_data = pd_pca_actor_means_emotion.merge(df_actor_emotions[["freebase_ID_actor", "actor_name"]], on="freebase_ID_actor", how="left").drop_duplicates(["freebase_ID_actor", "actor_name"])

    plot_3D_actor_emotion(df_data, colors_label, title="Most present emotion in actor's movie")

def plot_3D_actor_review_emotion(df_characters, df_reviews_with_emotions):
    """
    Given the dataframe containing the actors name along with the movie they played in and 
    the emotions of the reviews of the movies.
    Plot a 3D scatter of the actors according to their most present emotion in the reviews on the movies they played in.
    Displays the 3D scatter plot by using PCA on the emotions means.
    """
    df_reviews_with_emotions_normalized = normalize_review_emotions(df_reviews_with_emotions, with_neutral=False)[["wikipedia_ID", 
                                                                                        "normalized_review_anger_without_neutral", 
                                                                                        "normalized_review_disgust_without_neutral", 
                                                                                        "normalized_review_fear_without_neutral", 
                                                                                        "normalized_review_joy_without_neutral", 
                                                                                        "normalized_review_sadness_without_neutral", 
                                                                                        "normalized_review_surprise_without_neutral"]]
    df_actor_emotions = df_characters.merge(df_reviews_with_emotions_normalized, on="wikipedia_ID")

    actors_min_movies = df_actor_emotions.groupby("freebase_ID_actor")["wikipedia_ID"].count() >= MIN_REVIEWS_ACTOR
    df_actor_emotions = df_actor_emotions[df_actor_emotions["freebase_ID_actor"].isin(actors_min_movies[actors_min_movies].index)]

    df_actor_mean_emotions = df_actor_emotions.groupby("freebase_ID_actor").agg(mean_anger=("normalized_review_anger_without_neutral", "mean"), 
                                                    mean_disgust=("normalized_review_disgust_without_neutral", "mean"),
                                                    mean_fear=("normalized_review_fear_without_neutral", "mean"),
                                                    mean_joy=("normalized_review_joy_without_neutral", "mean"),
                                                    mean_sadness=("normalized_review_sadness_without_neutral", "mean"),
                                                    mean_surprise=("normalized_review_surprise_without_neutral", "mean"))
    
    colors_label = df_actor_mean_emotions.idxmax(axis=1).str.replace("mean_","")

    pca = PCA(n_components=3)
    pd_pca_actor_means_emotion = pd.DataFrame(pca.fit_transform(df_actor_mean_emotions), index=df_actor_mean_emotions.index).reset_index()

    # Merge data we want to display
    df_data = pd_pca_actor_means_emotion.merge(df_actor_emotions[["freebase_ID_actor", "actor_name"]], on="freebase_ID_actor", how="left").drop_duplicates(["freebase_ID_actor", "actor_name"])

    plot_3D_actor_emotion(df_data, colors_label, title="Most present emotion in actor's movie review")

def plot_3D_actor_emotion(df_data, colors_label, title):
    """
    Given some 3D coordinates along with the name of the actor (df_data) and the colors of the point.
    Display a 3D scatter with the given title and display as 'hover' action over the points the name of the actor associated to the point.
    """

    fig = px.scatter_3d(df_data, custom_data=['actor_name'], x=0, y=1, z=2, color=colors_label, width=750, height=750)

    # https://stackoverflow.com/questions/68960000/plotly-how-to-rename-legend-elements-of-a-plotly-express-stacked-bar-plot
    # Modified to modify the marker color
    fig.for_each_trace(lambda t: t.update(marker = dict(color=COLORS[t.name])))
    fig.for_each_trace(lambda t: t.update(hovertemplate="%{customdata[0]}<extra></extra>"))

    # Modify marker size
    fig.update_traces(marker_size = 4)

    # Modifer axis: https://stackoverflow.com/questions/73825017/turn-off-visibility-of-axis-and-grids-in-python-plotly-go-scatter3d
    fig.update_scenes(xaxis=dict(showgrid = False,showticklabels = False, title=" ", visible=False))
    fig.update_scenes(yaxis=dict(showgrid = False,showticklabels = False, title=" ", visible=False))
    fig.update_scenes(zaxis=dict(showgrid = False,showticklabels = False, title=" ", visible=False))

    # Modify legend
    fig.update_layout(legend_title_font_size = 20, legend_font_size=15, legend_title="Emotion")

    # Modify legend marker: https://community.plotly.com/t/modify-marker-size-in-legend/5338/12
    fig.update_layout(legend_itemsizing='constant')

    # Modify title
    fig.update_layout(title_text=title, title_x=0.5, title_font_size=25)
    fig.show()
