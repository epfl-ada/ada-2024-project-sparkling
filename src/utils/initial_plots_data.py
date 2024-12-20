import os
import nltk
import re
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import seaborn as sns
from collections import Counter
from nltk.stem import WordNetLemmatizer
from nltk.corpus import stopwords
from sklearn.feature_extraction.text import ENGLISH_STOP_WORDS

nltk.download("stopwords")
nltk.download('wordnet')

FONT_SIZE = 10
TITLE_FONT_SIZE = 15
COLORS = {
    'joy': '#f3c35f',
    'anger': '#e13c29',
    'disgust': '#8bc664',
    'sadness': '#85b8ed',
    'fear': '#b99be5',
    'surprise': '#de9f81'
}
EMOJIS = {
    'joy': '😂',
    'anger': '😡',
    'disgust': '🤢',
    'sadness': '😭',
    'fear': '😨',
    'surprise': '😯'
}

# List of emotions
EMOTIONS = ["anger", "disgust", "fear", "joy", "sadness", "surprise"]


def plot_movies_over_time_months(df_movies):
    """
    Plot the number of movies over the years (at a month precision)

    Argument:
        - df_movies: Dataframe containing the years and months of the movies releases
            (Columns: release_year and release_month)
    """
    plt.figure()

    years_months_series = (
        df_movies["release_year"].astype(str)
        + "-"
        + df_movies["release_month"].astype(str)
    )
    years_months_series = pd.to_datetime(years_months_series)
    sorted_bins_year_months = years_months_series.value_counts().sort_index()
    ax = sorted_bins_year_months.plot(
        figsize=(11, 5), xlabel="Time", ylabel="Number of movies"
    )
    ax.set_title("Movie release over time (month precision)", fontsize=TITLE_FONT_SIZE)

    # Force the first and last label to appear
    STEP_YEARS = 10
    first_date = sorted_bins_year_months.index[0]
    last_date = sorted_bins_year_months.index[-1]

    list_middle_labels_years = pd.Series(
        range(first_date.year + STEP_YEARS, last_date.year - STEP_YEARS + 1, STEP_YEARS)
    )
    list_middle_labels_dates = pd.to_datetime(
        list_middle_labels_years.to_list(), format="%Y"
    ).to_list()
    ax.set_xticks([first_date] + list_middle_labels_dates + [last_date])

    # https://matplotlib.org/stable/gallery/text_labels_and_annotations/date.html
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y-%b"))


def plot_movies_over_time_years(df_movies):
    """
    Plot the number of movies over the years (at a year precision)

    Argument:
        - df_movies: Dataframe containing the years and months of the movies releases
            (Columns: release_year and release_month)
    """
    plt.figure()

    sorted_bins_years = df_movies["release_year"].value_counts().sort_index()
    fig, ax = plt.subplots(figsize=(11, 7))
    sns.barplot(x=sorted_bins_years.index.astype(str), y=sorted_bins_years.values, ax=ax)
    
    ax.set_xticks(ax.get_xticks()[::10])
    
    ax.set_title("Movie release over time (year precision)", fontsize=TITLE_FONT_SIZE)
    ax.set_xlabel("Time")
    ax.set_ylabel("Number of movies")


def plot_pie_genres_proportion(df_genres):
    """
    Plot a pie chart given an one hot encoding of the genres per movies.

    Argument:
        - df_genres: One hot encoding of the genres for each movies
    """
    plt.figure(figsize=(12, 12))

    genres = df_genres
    if "wikipedia_ID" in df_genres.columns:
        genres = genres.drop("wikipedia_ID", axis=1)

    genres_counts = genres.sum().sort_values(ascending=False)

    # To rotate the genres and percentages:
    # https://stackoverflow.com/questions/64411633/how-to-rotate-the-percentage-label-in-a-pie-chart-to-match-the-category-label-ro
    patches, labels, pct_texts = plt.pie(
        genres_counts.values,
        labels=genres_counts.index,
        colors=sns.color_palette("colorblind"),
        autopct="%1.2f%%",
        pctdistance=0.9,
        rotatelabels=True
    )
    for label, pct_text in zip(labels, pct_texts):
        pct_text.set_rotation(label.get_rotation())
        pct_text.set_fontsize(FONT_SIZE)
        label.set_fontsize(FONT_SIZE)

    plt.title("Proportion of genres", fontsize=TITLE_FONT_SIZE, y=1.05)


def plot_donut_top_countries(df_countries, top=10, exclude_countries=None):
    plt.figure(figsize=(11, 7))

    countries = df_countries
    countries_list = df_countries.columns
    if "wikipedia_ID" in countries_list:
        countries = countries.drop("wikipedia_ID", axis=1)

    if exclude_countries is not None:
        for exclude_country in exclude_countries:
            if exclude_country in countries_list:
                countries = countries.drop(exclude_country, axis=1)

    countries_count = countries.sum().sort_values(ascending=False)

    # Consider all the others countries as "Others"
    total = countries_count.sum()
    serie_top_countries = countries_count[: (top - 1)]
    serie_top_countries["Others"] = total - serie_top_countries.sum()

    serie_top_countries = serie_top_countries.sort_values(ascending=False)

    plt.pie(
        serie_top_countries.values,
        labels=serie_top_countries.index,
        colors=sns.color_palette("colorblind"),
        autopct="%1.2f%%",
        pctdistance=0.85,
    )

    plt.title(f"Top {top} movie countries", fontsize=TITLE_FONT_SIZE, y=0.48)

    # Make a donut pie: https://proclusacademy.com/blog/customize_matplotlib_piechart/
    hole = plt.Circle((0, 0), 0.6, facecolor="white")
    plt.gcf().gca().add_artist(hole)


def show_top_10_words_per_emotion(
    df_movies_with_emotions_normalized, top_emotions=1, output_dir="emotion_word_plots"
):
    """
    Displays and saves bar plots of the top 10 words associated with each dominant emotion in movie plots.

    Args:
    df_movies_with_emotions (pd.DataFrame): DataFrame containing columns for normalized emotion scores and plots.
    top_emotions (int): Number of top emotions to consider per plot.
    output_dir (str): Directory path where plots will be saved.
    """

    # Define the emotions to analyze
    emotion_columns = [
        "normalized_plot_anger_without_neutral",
        "normalized_plot_joy_without_neutral",
        "normalized_plot_sadness_without_neutral",
        "normalized_plot_disgust_without_neutral",
        "normalized_plot_fear_without_neutral",
        "normalized_plot_surprise_without_neutral",
    ]

    # Identify top emotions per movie plot and store in a new column
    df_movies_with_emotions_normalized["top_emotions"] = df_movies_with_emotions_normalized[
        emotion_columns
    ].apply(lambda row: row.nlargest(top_emotions).index.tolist(), axis=1)

    # Initialize the lemmatizer and the list of stopwords
    lemmatizer = WordNetLemmatizer()
    stop_words = set(stopwords.words("english")).union(ENGLISH_STOP_WORDS)

    # Initialize counters for words associated with each emotion
    word_counts_by_emotion = {emotion: Counter() for emotion in emotion_columns}

    # Process each plot text to update word counts by associated top emotions
    for _, row in df_movies_with_emotions_normalized.iterrows():
        plot_text = row["plot"].lower()
        for emotion in row["top_emotions"]:
            if emotion in word_counts_by_emotion:
                # Tokenize, clean, and lemmatize words while removing stop words
                words = [
                    lemmatizer.lemmatize(word)
                    for word in re.findall(r"\b\w+\b", plot_text)
                    if word not in stop_words
                ]
                word_counts_by_emotion[emotion].update(words)

    # Ensure the output directory exists
    os.makedirs(output_dir, exist_ok=True)

    # Set up the plotting grid (2 rows x 3 columns)
    fig, axes = plt.subplots(2, 3, figsize=(20, 10))

    # Generate bar plots for the top 10 words in each emotion category
    for idx, (emotion, word_counts) in enumerate(word_counts_by_emotion.items()):
        common_words = word_counts.most_common(10)
        if common_words:
            words, counts = zip(*common_words)
            sns.barplot(x=list(words), y=list(counts), ax=axes[idx // 3, idx % 3])
            axes[idx // 3, idx % 3].set_title(
                f"Top 10 Words for {emotion.split('_')[2].capitalize()}"
            )

    # Finalize and save the plot
    plt.tight_layout()
    # plt.savefig(os.path.join(output_dir, 'top_words_per_emotion.png'))
    plt.show()

def plot_emotion_distribution_per_period_old(dataframe, period_column):

    """
    Generates a stacked bar chart of emotion distribution for each period : in our case we use it for months ans years.
    For each period (month or year), we split a 1-unit bar into segments representing the proportion of each emotion 
    within that period.

    Parameters:
    - dataframe (pd.DataFrame): DataFrame with normalized emotion scores (without neutral) and specified period column.
    - period_column (str): Column name for grouping data ('release_year' or 'release_month').

    Returns:
    - Stacked bar chart showing the distribution of each emotion per period.
    """
    
    emotion_labels = ["anger", "disgust", "fear", "joy", "sadness", "surprise"]
    emotions = [f'normalized_plot_{emotion}_without_neutral' for emotion in emotion_labels]
    
    # calculate the proportion of each emotion for each period
    period_emotion_totals = dataframe.groupby(period_column)[emotions].sum()
    period_emotion_distribution = period_emotion_totals.div(period_emotion_totals.sum(axis=1), axis=0)
    
    # Create the plot
    plt.figure(figsize=(14, 8))
    bottom = None  

    #One bar for each emotion
    for i, emotion in enumerate(emotions):
        emotion_label = emotion_labels[i]
        emotion_color = COLORS[emotion_label]
        if bottom is None:
            bottom = period_emotion_distribution[emotion]
            plt.bar(period_emotion_distribution.index, period_emotion_distribution[emotion], 
                    label=emotion_label, color=emotion_color)
        else:
            plt.bar(period_emotion_distribution.index, period_emotion_distribution[emotion], bottom=bottom, 
                    label=emotion_label, color=emotion_color)
            bottom += period_emotion_distribution[emotion]

    plt.title(f"Distribution of emotions per {period_column.replace('release_', '')}")
    plt.xlabel(period_column.replace('release_', '').capitalize())
    plt.ylabel("Proportion")
    plt.legend(title="Emotion", bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.show()

def plot_emotion_deviation_per_period(dataframe, reference_period='release_month'):

    """
    Plots the deviation of emotions from their overall mean across the specified period to observe fluctuations or trends 
    in emotion intensities over time. This visualization helps to identify periods where certain emotions are notably higher 
    or lower than their overall average, providing insights into potential seasonal variations in emotional content.
    
    Parameters:
    - dataframe (pd.DataFrame): DataFrame containing normalized emotion scores (without neutral) and the specified period column.
    - reference_period (str): The column name representing the period to group by ('release_month' or 'release_year').
    
    Returns:
    - Line plot showing the deviation of each emotion from its overall mean over the specified period.
    """

    emotion_labels = ["anger", "disgust", "fear", "joy", "sadness", "surprise"]
    emotions = [f'normalized_plot_{emotion}_without_neutral' for emotion in emotion_labels]
    
    overall_mean = dataframe[emotions].mean()
    period_data = dataframe.groupby(reference_period)[emotions].mean()

    deviation_from_mean = period_data - overall_mean

    #Plotting deviations
    plt.figure(figsize=(12, 6))
    for i, emotion in enumerate(emotions):
        emotion_label = emotion_labels[i]
        emotion_color = COLORS[emotion_label]
        plt.plot(deviation_from_mean.index, deviation_from_mean[emotion], label=emotion_label, color=emotion_color)
        
    plt.title("Index of deviation of emotions from the mean")
    plt.xlabel(reference_period.replace('release_', '').capitalize())
    plt.ylabel("Deviation from the mean")
    plt.legend(title="Emotion", bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.show()

def plot_major_emotion_per_five_years(dataframe, year_column='release_year'):

    """
    This function groups data by five year intervals and finds the dominant emotion 
    (the one with the highest average score) in each period. It then visualizes the 
    percentage of the dominant emotion per period.

    Parameters:
    - dataframe (pd.DataFrame): DataFrame containing normalized emotion scores (without neutral) and a release year column.
    - year_column (str): Column name for the release year (default is 'release_year').

    Returns:
    - Bar plot with the dominant emotion and its average percentage per period.
    """

    emotion_labels = ["anger", "disgust", "fear", "joy", "sadness", "surprise"]
    emotions = [f'normalized_plot_{emotion}_without_neutral' for emotion in emotion_labels]
    
    #Group the data by five year intervals, using the beginning year of each period
    dataframe['period'] = (dataframe[year_column] // 5) * 5
    
    #Calculate the mean emotion scores for each period
    period_means = dataframe.groupby('period')[emotions].mean()
    
    #Find the dominant emotion and its percentage for each period
    dominant_emotion = period_means.idxmax(axis=1).str.replace('normalized_plot_', '')
    dominant_emotion = dominant_emotion.str.replace('_without_neutral', '')
    dominant_percentage = period_means.max(axis=1)
    
    #Add dominant emotion and percentage as new columns to period_means
    period_means['Dominant Emotion'] = dominant_emotion
    period_means['Percentage'] = dominant_percentage

    #Plotting
    plt.figure(figsize=(14, 6))
    sns.barplot(x=period_means.index, y='Percentage', hue='Dominant Emotion', data=period_means, palette=COLORS)
    plt.title("Dominant emotion for each quinquennium")
    plt.xlabel("Five year period")
    plt.ylabel("Proportion")
    plt.legend(title="Emotion", bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.tight_layout()
    plt.show()