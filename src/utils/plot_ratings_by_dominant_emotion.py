import plotly.express as px

from src.utils.plot_countries_plots import get_most_dominant_emotion
from src.utils.initial_plots_data import COLORS
from src.utils.save_plot import save_plot

def plot_ratings_by_most_dominant_emotion(df_reviews_with_emotions):
    """
    Given a dataframe containing the reviews emotions
    plot the ratings by the most dominant emotion.

    Argument:
        - df_reviews_with_emotions: Dataframe containing the reviews emotions
    """
    reviews_with_dominant_emotions = get_most_dominant_emotion(df_reviews_with_emotions.drop('review_neutral', axis=1), reviews=True)

    fig = px.box(
        reviews_with_dominant_emotions, 
        x='dominant_emotion', 
        y='rating', 
        title='Ratings by dominant emotion',
        labels={'dominant_emotion': 'Dominant emotion', 'rating': 'Rating'},
        color='dominant_emotion',
        color_discrete_map=COLORS
    )

    fig.update_layout(
        plot_bgcolor='white',
        paper_bgcolor='white'
    )

    fig.show()

    save_plot(fig, "ratings_boxplot_final")