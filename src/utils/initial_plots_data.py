import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import seaborn as sns

FONT_SIZE = 12
TITLE_FONT_SIZE = 15


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
    sns.barplot(x=sorted_bins_years.index, y=sorted_bins_years.values, ax=ax)

    # Force the first and last label to appear
    first_date = sorted_bins_years.index[0]
    last_date = sorted_bins_years.index[-1]
    STEP_YEARS = 10

    list_labels_years = [
        str(x) for x in list(range(first_date, last_date + 1, STEP_YEARS))
    ]
    ax.set_xticks(list_labels_years, labels=list_labels_years)

    ax.set_title("Movie release over time (year precision)", fontsize=TITLE_FONT_SIZE)
    ax.set_xlabel("Time")
    ax.set_ylabel("Number of movies")


def plot_pie_genres_proportion(df_genres):
    """
    Plot a pie chart given an one hot encoding of the genres per movies.

    Argument:
        - df_genres: One hot encoding of the genres for each movies
    """
    plt.figure(figsize=(11, 7))

    genres = df_genres
    if "wikipedia_ID" in df_genres.columns:
        genres = genres.drop("wikipedia_ID", axis=1)

    genres_counts = genres.sum().sort_values(ascending=False)

    # To rotate the genres and percetange:
    # https://stackoverflow.com/questions/64411633/how-to-rotate-the-percentage-label-in-a-pie-chart-to-match-the-category-label-ro
    patches, labels, pct_texts = plt.pie(
        genres_counts.values,
        labels=genres_counts.index,
        colors=sns.color_palette("colorblind"),
        autopct="%1.2f%%",
        pctdistance=0.8,
        rotatelabels=True,
    )
    for label, pct_text in zip(labels, pct_texts):
        pct_text.set_rotation(label.get_rotation())
        pct_text.set_fontsize(FONT_SIZE)
        label.set_fontsize(FONT_SIZE)

    plt.title("Proportion of genres", fontsize=TITLE_FONT_SIZE, y=1.2)


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
