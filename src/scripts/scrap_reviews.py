import time
from selenium import webdriver
from selenium.webdriver.common.by import By
from bs4 import BeautifulSoup
from selenium.webdriver.support import expected_conditions as EC
import pandas as pd


def scrape_imdb_reviews(movie_id):
    """
    Scrap all the reviews of a movie given an IMDb id (tconst)

    Args :
    movie_id : the IMDb id of the movie we want information about

    Returns: 
    a list of reviews, stars and whether the review
    is a spoiler as three seperated lists
    """

    #URL for the movie's reviews page on IMDb
    url = f"https://www.imdb.com/title/{movie_id}/reviews"
    print(movie_id)

    driver = webdriver.Chrome()
    driver.get(url)
    #sleep to wait for the page to load
    time.sleep(2)

    try:
        buttons = driver.find_elements(By.CLASS_NAME, "ipc-btn")

        #expanding the page if more reviews have to be loaded
        tout_counter = 0
        for button in buttons:
            if button.text.strip() == "Tout":
                tout_counter += 1
            if tout_counter == 2:
                driver.execute_script(
                    "arguments[0].scrollIntoView();", button
                )  # Scroll to the button to ensure it's visible
                time.sleep(2)
                button.click()
                break

        # to ensure the remaining reviews load
        time.sleep(10)

        #getting all the spoilers' buttons to reveal spoiling reviews
        js_script = """
                const buttons = Array.from(document.querySelectorAll('span.ipc-btn__text'));
                buttons.forEach(button => {
                if (button.textContent.trim() === 'Spoiler') {
                    button.click();
                }
            });
        """
        driver.execute_script(js_script)

        html_from_page = driver.page_source

        soup = BeautifulSoup(html_from_page, "html.parser")

        review_containers = soup.find_all("div", class_="ipc-list-card__content")
        reviews = []
        #a score given by the reviewer, can be between 0 and 10.
        stars = []
        #1 if the review is a spoiler, 0 otherwise
        spoilers = []
        for container in review_containers:
            review_div = container.find("div", class_="ipc-html-content-inner-div")
            review_text = (
                review_div.get_text(strip=True) if review_div else "No review available"
            )

            star_rating_tag = container.find("span", class_="ipc-rating-star--rating")
            if star_rating_tag:
                star_rating = star_rating_tag.get_text(strip=True)
            else:
                star_rating = None

            spoiler_tag = container.find("div", class_="ipc-signpost__text")
            spoiler = 0
            if spoiler_tag and spoiler_tag.get_text(strip=True) == "Spoiler":
                spoiler = 1

            reviews.append(review_text)
            stars.append(star_rating)
            spoilers.append(spoiler)

    finally:
        driver.quit()

    return reviews, stars, spoilers


def scrape_reviews(df, filename="scrapped_imdb_reviews.csv"):
    """
    Scrap all the reviews, stars and spoiler tags of all movies in the given dataframe

    Args : 
    df : dataframe containing the movies. It must contain a row called 'tconst' with the movies' IMDb IDs
    filename : the csv file to store the movies with their new information in

    Returns: 
    None
    """
    expanded_rows = []
    for index, row in df.iterrows():
        tconst = row["tconst"]
        reviews, stars, spoilers = scrape_imdb_reviews(tconst)

        for review, star, spoiler_tag in zip(reviews, stars, spoilers):
            new_row = row.tolist() + [review, star, spoiler_tag]
            expanded_rows.append(new_row)

    expanded_df_next = pd.DataFrame(
        expanded_rows,
        columns=list(df.columns) + ["review_summary", "rating", "spoiler_tag"],
    )

    expanded_df_next.to_csv(filename, index=False)
