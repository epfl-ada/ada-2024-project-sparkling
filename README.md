# SPARKling

## Installation
```
conda create --name <env_name> python=3.11.9
conda activate <env_name>
pip install -r pip_requirements.txt
```


## Project proposal 

### From screen to heart : how do emotions shape the cinematic landscape ?

#### Abstract

Movies have a unique power to immerse us, that we feel as if we're living in the characters' shoes. They transport our minds, allowing us to get attached or disappointed. Sometimes, it even happens that we jump out of scareness in front of a horror movie or cry because our favorite hero has just died. Inspired by this captivating influence, we have decided to delve deep into an emotional analysis around movies and focus our data story on showing how emotions are transmitted to the viewer throughout a movie, how it differs around the world and how the emotions are related to one another. Then we are going to extend our analysis by looking at the reviews posted by the viewers and relate their emotions to those of the movie.

#### Research questions

* What are the main emotions conveyed by the movie we watch, and how does this change around the world? This helped us get some insights about how the different cultures around the world relate to emotions.  
* How do emotions relate to the movies’ genres? We caught the emotional patterns across the genres.  
* Is there some correlation between the month of the release of a movie and the main emotions conveyed ? Through this question, we wanted to know if, for instance, scary movies are always released around Halloween.  
* Is there an emotion that prevails over the years through the released movies ? Using this question, we uncovered emotional trends over the years.  
* Is there some actor that is always associated with certain emotions ?  
* For each of the questions above, we also uncovered the question for reviews emotions instead of the movie plots emotions, and compared the diverse results.  
* Can we see some patterns in the progression of emotions throughout a movie? This helped us have an idea of the correlation between the genre of the movie and the emotions that we are expected to feel while watching them.


#### Data preparation

Extending our base dataset :  
We decided to combine the CMU movie summary corpus which originally contained 42’303 movies, with two different datasets. First we merged our movie dataset with an IMDB reviews dataset from Kaggle ([https://www.kaggle.com/datasets/ebiswas/imdb-review-dataset/data](https://www.kaggle.com/datasets/ebiswas/imdb-review-dataset/data)) allowing us to have 1'859'786 different reviews associated with our movies. Then, we used the non-commercial IMDB dataset ([https://developer.imdb.com/non-commercial-datasets](https://developer.imdb.com/non-commercial-datasets)), to associate our movies with their IMDB ID, and we extended our reviews and scores dataset by scraping 101’320 more reviews from IMDB. Since the datasets from IMDB contained a movie+year field, we matched our reviews datasets and original dataset based on their name and year of publication. Moreover, some movies provided by our base dataset did not have any year or month of publication. So we tried to complete them by scraping them from Wikidata based on their Wikipedia ID.  
Creation of the emotional dataset:  
Naturally, the most important aspect of our dataset is the emotional part, containing the emotions of the movies’ plots and reviews. We have considered different models for the emotions calculation before adapting the Emotion English DistilRoberta model from HuggingFace: [https://huggingface.co/j-hartmann/emotion-english-distilroberta-base](https://huggingface.co/j-hartmann/emotion-english-distilroberta-base). This model extracts the emotions of anger, disgust, fear, joy, neutral, sadness, and surprise from a text input. Our choice was mainly influenced by the emotions this model is able to extract, the used datasets for its training as well as its number of downloads.  

#### Preprocessing 
For our final analysis, we decided to filter our data even further to keep only the most valuable data. First, we removed the columns that we will not need such as the revenues. Then, we noticed that movie plots with a few sentences did not have an end-to-end detailed description of the plot of the movie. Therefore, we decided to keep the movies with more than 10 sentences to make our emotion analysis richer. We also decided to keep only movies that had at least 10 reviews to once again have a more precise emotional analysis of the reviews. We, however, decided to restrain to a maximum of 50 reviews per movie when there were more, by a random selection, to prevent excessive computation while keeping 11’564 movies satisfying all our requirements in the end. And since there are 364 genres involved, we decided to consolidate them under 22 main genres, following Wikipedia's categorization. This allows us to group related sub-genres, making our analysis more focused and meaningful. So our final dataset is composed of 11’564 movies and 449’824 user reviews.

#### High-level methods

We used the K-means algorithm To split our movies into 2 clusters based on the plots and the reviews. We also measured the correlation between the pairs of clusters using Spearman and Pearson correlation.  
The latter was also used to compute the linear relation between the movies’ genres and emotions.  
In order to display the actors in compliance with the mean emotions of the movies they played in, across the plots and also the reviews, we used dimensionality reduction through PCA.  
With the aim of analyzing the different progression of emotions throughout a movie, we look at the transition distribution and variations between the different emotions throughout the plot of the movie. These transition probabilities helped us to discover some patterns in the movie emotional profile.   
Finally, we developed a movie recommender based on emotions. By assigning each movie a score across various emotions, the recommender will suggest films that best match the emotional experience users are seeking, based on a pre-calculated json file across all possibilities.

#### Contribution

- Sofia Taouhid : Scraping of released dates from Wikidata, plots and analysis of emotion transitions throughout movies, datastory  
- Matthias Wyss : Ran model of emotions on plots and reviews, plots and analysis of genres and emotions, website creation  
- Lina Sadgal : Scraping of reviews from IMDb, plots and analysis of countries and emotions, recommender system, datastory  
- Yasmine Tligui : Extraction of ratings from IMDb, correlation between genres and emotions, emotion trends over year periods, datastory  
- Guillaume Vitalis : Data cleaning, initial dataset analysis, clustering of movies, emotional profile of actors, code merging

#### Timeline <br>
Week 1 - 25.11 : First analysis and visualizations to answer our high-level questions <br>
Week 2 - 2.12 : Drawing of correlations between previous analysis <br>
Week 3 - 9.12 : Working on a storyline for the website <br>
Week 4 -  16.12 : Finalizing the website and report writing <br>



## Project structure

```
├── data                          <- Project data files
│   ├── FinalData                 <- Our scraped data
│   │   ├── final_dates.csv
│   │   ├── scrapped_imdb_reviews.csv
│   │   ├── final_reviews.csv     <- Created with all the other datasets
│   │   └── final_movies.csv      <- Created with all the other datasets
│   ├── IMDB                      <- https://developer.imdb.com/non-commercial-datasets/
│   │   ├── title.basics.tsv
│   │   └── title.ratings.tsv
│   ├── IMDBReviewsKaggle         <- https://www.kaggle.com/datasets/ebiswas/imdb-review-dataset/data
│   │   ├── IMDBReviewsKaggle.csv <- CSV merging all the parts and keeping only the interesting reviews
│   │   ├── part-01.json
│   │   ├── part-02.json
│   │   ├── part-03.json
│   │   ├── part-04.json
│   │   ├── part-05.json
│   │   ├── part-06.json
│   │   └── sample.json           <- Not used
│   ├── MovieSummaries            <- Additional movie data
│   |   ├── README.txt
│   |   ├── character.metadata.tsv
│   |   ├── movie.metadata.tsv
│   |   ├── name.clusters.txt
│   |   ├── plot_summaries.txt
│   |   └── tvtropes.clusters.txt
|   └── PredictedEmotions         <- Our emotional data
│       ├── plot_emotions.tsv     <- scores emotion of the plots
|       └── review_emotions.tsv   <- scores emotion of the users' reviews
│
├── src                           <- Source code
│   ├── data                      <- Data directory with methods related to it
│   ├── models                    <- Model directory
│   ├── utils                     <- Utility directory & plots
│   └── scripts                   <- Additional scripts
│                        
│
├── results.ipynb                 <- A well-structured notebook showing the results
│
├── .gitignore                    <- List of files ignored by git
├── pip_requirements.txt          <- File for installing Python dependencies
└── README.md                     <- Project overview and documentation
```
