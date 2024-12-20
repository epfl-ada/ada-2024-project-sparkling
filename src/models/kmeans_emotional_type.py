import matplotlib.pyplot as plt
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score

def get_best_k_clustered_movie_emotional_type(df_emotions_normalized, is_review=False, min_k=2, max_k=10):
    """
    Given the emotions of a movie (either the plot or the reviews) along with the range of k we want to run.
    Compute print and plot the silhouette of the KMeans algorithm on the given data and return the best k

    Arguments:
        - df_emotions_normalized: Emotions of the movie (either for plots or reviews emotions) normalized 
            without the neutral emotion
        - is_review: Whether the df_emotions_normalized is containing the reviews emotions
        - min_k, max_k: range of k to run KMeans on.

    Returns the best k found
    """    
    if not is_review:
        df_movies_normalized_emotions = df_emotions_normalized[["wikipedia_ID", "normalized_plot_anger_without_neutral", 
                                                                                    "normalized_plot_disgust_without_neutral", 
                                                                                    "normalized_plot_fear_without_neutral", 
                                                                                    "normalized_plot_joy_without_neutral", 
                                                                                    "normalized_plot_sadness_without_neutral", 
                                                                                    "normalized_plot_surprise_without_neutral"]]
        dataX = df_movies_normalized_emotions.drop("wikipedia_ID", axis=1)

    else:
        df_movies_reviews_normalized_emotions = df_emotions_normalized[["wikipedia_ID","normalized_review_anger_without_neutral", 
                                                                                    "normalized_review_disgust_without_neutral", 
                                                                                    "normalized_review_fear_without_neutral", 
                                                                                    "normalized_review_joy_without_neutral", 
                                                                                    "normalized_review_sadness_without_neutral", 
                                                                                    "normalized_review_surprise_without_neutral"]]
        
        dataX = df_movies_reviews_normalized_emotions.groupby("wikipedia_ID").mean()

    # From exercise 8 solution

    # Get the silhouettes score
    silhouettes = []
    for k in range(min_k, max_k+1):
        labels = KMeans(n_clusters=k, random_state=10).fit_predict(dataX)

        score = silhouette_score(dataX, labels)

        silhouettes.append({"k": k, "score": score})
        

    # Convert to dataframe
    silhouettes = pd.DataFrame(silhouettes).set_index('k')

    # Plot the data
    plt.figure(figsize=(5, 5))
    plt.plot(silhouettes.index, silhouettes.score)
    plt.xlabel("K")
    plt.ylabel("Silhouette score")
    plt.title("Silhouette score on movies " + ("reviews" if is_review else "plot") + " emotions")
    plt.show()

    best_k = silhouettes.idxmax().iloc[0]
    print("Silhouettes: ")
    print(silhouettes)

    print("The best k is " + str(best_k))
    return best_k