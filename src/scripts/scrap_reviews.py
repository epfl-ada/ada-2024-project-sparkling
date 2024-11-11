import time
from selenium import webdriver
from selenium.webdriver.common.by import By
from bs4 import BeautifulSoup
from selenium.webdriver.support import expected_conditions as EC
import pandas as pd


def scrape_imdb_reviews(movie_id):
    """
    Scrap all the reviews of a movie given an IMDB id (tconst)

    Returns a list of reviews, stars and whether the review
    is a spoiler as three seperated lists
    """

    print("scrapping for id " + str(movie_id))
    # URL for the movie's reviews page on IMDb
    url = f"https://www.imdb.com/title/{movie_id}/reviews"

    driver = webdriver.Chrome()
    driver.get(url)
    # to wait for the page to load
    time.sleep(2)

    try:
        buttons = driver.find_elements(By.CLASS_NAME, "ipc-btn")

        tout_counter = 0
        for button in buttons:
            if button.text.strip() == "Tout":
                tout_counter += 1
            if tout_counter == 2:
                print("activating button Tout")
                driver.execute_script(
                    "arguments[0].scrollIntoView();", button
                )  # Scroll to the button to ensure it's visible
                time.sleep(2)
                button.click()
                break

        # to ensure the reviews load
        time.sleep(10)

        # script to get all the spoilers quickly
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
        stars = []
        # 0 if not, 1 otherwise
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
