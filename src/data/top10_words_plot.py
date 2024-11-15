import os
import matplotlib.pyplot as plt
import seaborn as sns
from collections import Counter
import nltk
from nltk.stem import WordNetLemmatizer
from nltk.corpus import stopwords
from sklearn.feature_extraction.text import ENGLISH_STOP_WORDS
import re

nltk.download('stopwords')
nltk.download('wordnet')

def show_top_10_words_per_emotion(df_movies_with_emotions_normalized, top_emotions=1, output_dir='emotion_word_plots'):
    """
    Displays and saves bar plots of the top 10 words associated with each dominant emotion in movie plots.

    Args:
    df_movies_with_emotions_normalized (pd.DataFrame): DataFrame containing columns for normalized emotion scores and plots.
    top_emotions (int): Number of top emotions to consider per plot.
    output_dir (str): Directory path where plots will be saved.
    """

    # Define the emotions to analyze
    emotion_columns = [
        'normalized_plot_anger_without_neutral', 'normalized_plot_joy_without_neutral', 'normalized_plot_sadness_without_neutral',
        'normalized_plot_disgust_without_neutral', 'normalized_plot_fear_without_neutral', 'normalized_plot_surprise_without_neutral'
    ]

    # Identify top emotions per movie plot and store in a new column
    df_movies_with_emotions_normalized['top_emotions'] = df_movies_with_emotions_normalized[emotion_columns].apply(
        lambda row: row.nlargest(top_emotions).index.tolist(), axis=1
    )

    # Initialize the lemmatizer and the list of stopwords
    lemmatizer = WordNetLemmatizer()
    stop_words = set(stopwords.words('english')).union(ENGLISH_STOP_WORDS)

    # Initialize counters for words associated with each emotion
    word_counts_by_emotion = {emotion: Counter() for emotion in emotion_columns}

    # Process each plot text to update word counts by associated top emotions
    for _, row in df_movies_with_emotions_normalized.iterrows():
        plot_text = row['plot'].lower()
        for emotion in row['top_emotions']:
            if emotion in word_counts_by_emotion:
                # Tokenize, clean, and lemmatize words while removing stop words
                words = [
                    lemmatizer.lemmatize(word)
                    for word in re.findall(r'\b\w+\b', plot_text)
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
            axes[idx // 3, idx % 3].set_title(f"Top 10 Words for {emotion.split('_')[2].capitalize()}")
            axes[idx // 3, idx % 3].set_xticklabels(words, rotation=45)

    # Finalize and save the plot
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'top_words_per_emotion.png'))
    plt.show()
