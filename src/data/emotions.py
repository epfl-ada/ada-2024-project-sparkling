import pandas as pd
from transformers import pipeline
from tqdm import tqdm
import re
import os
import matplotlib.pyplot as plt
import seaborn as sns
from collections import Counter
from scipy.cluster.hierarchy import linkage, dendrogram

import nltk
from nltk.stem import WordNetLemmatizer
nltk.download('stopwords')
from nltk.corpus import stopwords
from sklearn.feature_extraction.text import ENGLISH_STOP_WORDS


# Define colors for each emotion
COLORS = {
    'joy': '#f3c35f',
    'anger': '#e13c29',
    'disgust': '#8bc664',
    'sadness': '#85b8ed',
    'fear': '#b99be5'
}



# Predicts the emotions present in a given movie plot by using a pre-trained classifier.
def predict_emotions_from_plot(classifier, plot: str):
    plot = plot.strip()
    plot = re.sub(r'<.*?>', '', plot)
    plot = plot.replace('&ndash;', '–')
    
    predictions = {
        'anger': 0,
        'disgust': 0,
        'fear': 0,
        'joy': 0,
        'neutral': 0,
        'sadness': 0,
        'surprise': 0
    }
    
    if plot.strip() == "":
        return {label: None for label in predictions}
    
    sentences = plot.split('.')
    if sentences[-1].strip() == "":
        sentences = sentences[:-1]
    
    for s in sentences:
        while len(s) > 1500:
            sentences.append(s[1500:])
            s = s[:1500]
        s = s.strip()
        results = classifier(s)
        if not results:
            continue
            
        results = results[0]
        
        for result in results:
            label = result['label']
            score = result['score']
            predictions[label] += score
            
    total_score = sum(predictions.values())
    
    if total_score == 0:
        normalized_predictions = {label: None for label in predictions}
    else:
        normalized_predictions = {label: score / total_score for label, score in predictions.items()}
        
    return normalized_predictions



# Adds emotion prediction columns to a DataFrame of movies based on their plots and save them into a csv.
def predict_emotions_to_csv(df_movies, csv_path='emotions.csv'):
    classifier = pipeline("text-classification", model="j-hartmann/emotion-english-distilroberta-base", top_k=None)
    tqdm.pandas(desc="Processing emotions")
    emotion_predictions = df_movies['plot'].progress_apply(lambda plot: predict_emotions_from_plot(classifier, plot))
    df_emotion = pd.DataFrame(emotion_predictions.tolist())
    df_movies_with_emotions = pd.concat([df_movies, df_emotion.set_index(df_movies.index)], axis=1)
    
    df_movies_with_emotions.to_csv(csv_path,
                               columns=['wikipedia_ID', 'anger', 'disgust', 'fear', 'joy', 'neutral', 'sadness', 'surprise'], 
                               index=False)


# Function to merge the emotion predictions CSV with df_movies on wikipedia_ID
def merge_df_movies_with_emotions_csv(df_movies, csv_path='emotions.csv'):
    df_emotions = pd.read_csv(csv_path)
    merged_df = pd.merge(df_movies, df_emotions, on='wikipedia_ID')
    return merged_df


def show_top_10_words_per_emotion(df_movies_with_emotions, top_emotions=1, output_dir='emotion_word_plots'):
    # List of emotions
    emotion_columns = ['anger', 'joy', 'sadness', 'disgust', 'fear']
    
    # Drop rows with NaN values in the emotion columns
    df_movies_with_emotions = df_movies_with_emotions.dropna(subset=emotion_columns).copy()

    # Compute and store the top emotions for each movie
    df_movies_with_emotions['top_emotions'] = df_movies_with_emotions[emotion_columns].apply(lambda row: row.nlargest(top_emotions).index.tolist(), axis=1)
    
    # Initialize the lemmatizer
    lemmatizer = WordNetLemmatizer()

    # Get the list of English stop words
    stop_words = set(stopwords.words('english')).union(ENGLISH_STOP_WORDS)

    # Initialize a word counter for each emotion
    word_counts_by_emotion = {emotion: Counter() for emotion in emotion_columns}

    # Count words by dominant emotions
    for _, row in df_movies_with_emotions.iterrows():
        plot_text = row['plot'].lower()
        top_emotions = row['top_emotions']
        for top_emotion in top_emotions:
            if top_emotion in word_counts_by_emotion:
                # Clean text, remove stop words, and lemmatize words
                words = [
                    lemmatizer.lemmatize(word) 
                    for word in re.findall(r'\b\w+\b', plot_text) 
                    if word not in stop_words
                ]
                word_counts_by_emotion[top_emotion].update(words)

    # Create directory to save images
    os.makedirs(output_dir, exist_ok=True)

    # Set up the figure with 2 rows and 3 columns
    num_col = 3
    fig, axes = plt.subplots(2, num_col, figsize=(20, 10))
        
    # Plot the 10 most common words for each emotion
    for idx, (emotion, word_counts) in enumerate(word_counts_by_emotion.items()):
        common_words = word_counts.most_common(10)
        if common_words:
            words, counts = zip(*common_words)
            sns.barplot(x=list(words), y=list(counts), ax=axes[idx // num_col, idx % num_col])
            axes[idx // num_col, idx % num_col].set_title(f"Top 10 Words for {emotion}")
            axes[idx // num_col, idx % num_col].set_xticklabels(words, rotation=45)

    # Adjust layout and save
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'top_words_per_emotion.png'))
    plt.show()

    
    
def plot_emotions(df_movies_with_emotions, categories, category_type='genre'):
    # Remove duplicated columns in the DataFrame
    df_movies_with_emotions = df_movies_with_emotions.loc[:, ~df_movies_with_emotions.columns.duplicated()]

    # Initialize a dictionary to store average emotion proportions
    emotion_means = {}

    # Check if categories is a dict or a list
    if isinstance(categories, dict):
        # If it's a dictionary, iterate through keys (main categories)
        for main_category, subcategories in categories.items():
            # Initialize a cumulative emotion sums
            cumulative_emotion_sums = pd.Series(0, index=['joy', 'sadness', 'anger', 'disgust', 'fear'])
            for category in subcategories:
                if category in df_movies_with_emotions.columns:
                    # Select rows where the category column is `True`
                    category_df = df_movies_with_emotions[df_movies_with_emotions[category] == True]
                    # Calculate the sum of emotions excluding 'neutral'
                    emotion_sums = category_df[['joy', 'sadness', 'anger', 'disgust', 'fear']].sum()
                    cumulative_emotion_sums += emotion_sums
            
            # Normalize the cumulative emotion scores
            if cumulative_emotion_sums.sum() > 0:
                normalized_emotions = cumulative_emotion_sums / cumulative_emotion_sums.sum()
            else:
                normalized_emotions = cumulative_emotion_sums  # Keep original values if sum is zero

            # Store the normalized values in the dictionary
            emotion_means[main_category] = normalized_emotions
            
    elif isinstance(categories, list):
        # If it's a list, iterate through each category
        for category in categories:
            if category in df_movies_with_emotions.columns:
                # Select rows where the category column is `True`
                category_df = df_movies_with_emotions[df_movies_with_emotions[category] == True]
                # Calculate the sum of emotions excluding 'neutral'
                emotion_sums = category_df[['joy', 'sadness', 'anger', 'disgust', 'fear']].sum()
                
                # Normalize the emotion scores
                if emotion_sums.sum() > 0:
                    normalized_emotions = emotion_sums / emotion_sums.sum()
                else:
                    normalized_emotions = emotion_sums  # Keep original values if sum is zero

                # Store the normalized values in the dictionary
                emotion_means[category] = normalized_emotions
    else:
        raise ValueError("Le paramètre 'categories' doit être un dictionnaire ou une liste.")

    # Convert the dictionary to a DataFrame for easier plotting
    emotion_df = pd.DataFrame(emotion_means)

    # Transpose so categories are the rows and emotions are columns
    emotion_df = emotion_df.T

    # Plotting with specified colors
    emotion_df.plot(kind='bar', stacked=True, figsize=(12, 8), color=[COLORS[emotion] for emotion in emotion_df.columns])
    
    plt.title(f'Proportion of Emotions by {category_type.capitalize()}')
    plt.xlabel(category_type.capitalize())
    plt.ylabel('Emotion Score')
    plt.legend(title='Emotions')
    plt.tight_layout()
    plt.show()
    
    # Calculate and print top 10 genres or countries by each emotion
    for emotion in ['joy', 'sadness', 'anger', 'disgust', 'fear']:
        # Sort by emotion value and get top 10
        top_emotions = emotion_df[emotion].nlargest(10)
        print(f'Top 10 {category_type} for {emotion.capitalize()}:')
        print(top_emotions)
        print()  # Blank line for readability
        
    