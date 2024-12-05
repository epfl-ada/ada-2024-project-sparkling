import plotly.graph_objects as go
import pandas as pd
from sklearn.cluster import KMeans
from plotly.subplots import make_subplots
from src.data.merge_genres import get_genres_merged
from src.utils.initial_plots_data import COLORS

def plot_clustered_movie_emotional_type(df_emotions_normalized, df_genres, is_review=False, k=2, clusters_color = {0: "palevioletred", 1: "steelblue"}, clusters_name = {0: "Cluster 0", 1: "Cluster 1"}, clusters_col_subplot = {0:1, 1:2}):
    """
    Given the normalized emotions associated to movie plot or movie reviews along with the dataframe containing the genres of the movies, 
    cluster in k clusters using Kmeans and displays the mean movie genres contained in the clusters.

    Arguments:
        - df_emotions_normalized: Dataframe containing the emotions for each movies plots or each movies reviews
        - df_genres: Genres of the movies
        - is_review: Whether df_emotions_normalized contains the emotions of the plots or the reviews
        - k: Number of clusters
        - clusters_color: dictionnary containing a mapping: cluster -> color for the final barplots
        - clusters_name: dictionnary containing a mapping: cluster -> cluster name
        - clusters_col_subplot: dictionnary containing a mapping: cluster -> column display the barplot in the subplots.

    Returns:
        - The mean genres per clusters dataframe
        - The centroids of the clusters
    """
    if not is_review:
        df_movies_normalized_emotions = df_emotions_normalized[["wikipedia_ID", "normalized_plot_anger_without_neutral", 
                                                                                    "normalized_plot_disgust_without_neutral", 
                                                                                    "normalized_plot_fear_without_neutral", 
                                                                                    "normalized_plot_joy_without_neutral", 
                                                                                    "normalized_plot_sadness_without_neutral", 
                                                                                    "normalized_plot_surprise_without_neutral"]]
        dataX = df_movies_normalized_emotions.drop("wikipedia_ID", axis=1)
        data_wikipedia_id = df_movies_normalized_emotions["wikipedia_ID"]
        emotions = df_movies_normalized_emotions.columns.str.replace("normalized_plot_", "").str.replace("_without_neutral", "").drop("wikipedia_ID")

    else:
        df_movies_reviews_normalized_emotions = df_emotions_normalized[["wikipedia_ID","normalized_review_anger_without_neutral", 
                                                                                    "normalized_review_disgust_without_neutral", 
                                                                                    "normalized_review_fear_without_neutral", 
                                                                                    "normalized_review_joy_without_neutral", 
                                                                                    "normalized_review_sadness_without_neutral", 
                                                                                    "normalized_review_surprise_without_neutral"]]
        
        dataX = df_movies_reviews_normalized_emotions.groupby("wikipedia_ID").mean()
        data_wikipedia_id = dataX.index
        emotions = df_movies_reviews_normalized_emotions.columns.str.replace("normalized_review_", "").str.replace("_without_neutral", "").drop("wikipedia_ID")


    # Run Kmeans
    bestKmeans = KMeans(n_clusters=k, random_state=10).fit(dataX)
    clusters = pd.Series(bestKmeans.predict(dataX), index=data_wikipedia_id).reset_index().rename({0:"cluster"}, axis=1)

    # Get main genres
    df_main_genres = get_genres_merged(df_genres)
    df_main_genres
    merged_cluster_genre = df_main_genres.merge(clusters, on="wikipedia_ID").drop("wikipedia_ID", axis=1)

    genre_mean_per_cluster = merged_cluster_genre.groupby("cluster").mean().reset_index()
    
    subplot_titles = []
    for cluster in genre_mean_per_cluster.index:
        subplot_titles.append(f"Percentage of movie genre in cluster {clusters_name[cluster]} based on " + ("review" if is_review else "plot") + " emotions")
    fig = make_subplots(rows=1, cols=k, subplot_titles=subplot_titles)
    
    # Display each clusters
    for cluster in genre_mean_per_cluster.index:
        color_discrete_sequence = [clusters_color[cluster]]*genre_mean_per_cluster.shape[1]
        data_to_display = (genre_mean_per_cluster[genre_mean_per_cluster["cluster"] == cluster] * 100).round(2)
        data_to_display = data_to_display.melt(id_vars=["cluster"])

        data_to_display.columns = ["cluster", "Genre", "Percentage (%)"]
        row = 1
        col = clusters_col_subplot[cluster]
        fig.add_trace(go.Bar(x=data_to_display["Genre"].values, 
                             y=data_to_display["Percentage (%)"].values, 
                             hovertemplate="Genre: %{x}<br>Percentage: %{y} %<extra></extra>",
                             showlegend=False), 
                      row=row, 
                      col=col)
        
        fig.update_traces(marker_color=color_discrete_sequence, row=row, col=col)

        ticks = [10, 20, 30, 40, 50, 60, 70, 80]
        fig.update_yaxes(tickmode='array', tickvals = ticks, ticktext = [f"{x}% " for x in ticks])

        fig.update_yaxes(range=[0,80])

    fig.show()
    centroids = bestKmeans.cluster_centers_
    centroids = pd.DataFrame(centroids, index=genre_mean_per_cluster.index, columns=emotions)
    return genre_mean_per_cluster, centroids

def plot_emotions_centroids(df_centroids, clusters_name = {0: "Cluster 0", 1: "Cluster 1"}, clusters_col_subplot = {0:1, 1:2} , is_reviews=False):
    """
    Given the clusters emotions centroids, plot as bar plots the emotions of the centroids

    Arguments:
        - df_centroids: Dataframe containing the emotions centroids
        - clusters_name: dictionnary containing a mapping: cluster -> cluster name
        - clusters_col_subplot: dictionnary containing a mapping: cluster -> column display the barplot in the subplots.
        - is_reviews: Whether the emotions contained are for the plots or the reviews
    """
    k = len(df_centroids)

    subplot_titles = []
    for cluster in df_centroids.index:
        subplot_titles.append(f"Centroid of the {clusters_name[cluster]} for the movies " + ("reviews" if is_reviews else "plot") + " emotions")

    fig = make_subplots(rows=1, cols=k, subplot_titles=subplot_titles)

    for i, cluster in enumerate(df_centroids.index):
        fig.add_trace(
            go.Bar(x=df_centroids.columns.str.capitalize(), 
                   y=(df_centroids.loc[cluster]*100).round(2), 
                   showlegend=False,
                   hovertemplate="%{x}: %{y} %<extra></extra>"), row=1, col=clusters_col_subplot[i]
        )
    fig.update_traces(marker_color=df_centroids.columns.map(COLORS))


    max_y_range = int(df_centroids.max().max() * 100) + 5
    ticks = list(range(5, max_y_range, 5))
    fig.update_yaxes(tickmode='array', tickvals = ticks, ticktext = [f"{x}% " for x in ticks])

    y_range = [0,max_y_range]
    fig.update_yaxes(range=y_range, title="Mean percentage", row=1, col=1)
    fig.update_yaxes(range=y_range, row=1, col=2)
    fig.update_xaxes(title_text="Genre")


    fig.show()