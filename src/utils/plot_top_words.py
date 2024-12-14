import os
import re
import numpy as np
from PIL import Image
from collections import Counter
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from sklearn.feature_extraction.text import ENGLISH_STOP_WORDS
from wordcloud import WordCloud
import plotly.graph_objects as go
import matplotlib.pyplot as plt
from src.utils.plot_genres import EMOTIONS, COLORS

PLOTS_PATH = "plots"

# Color mapping for emotions in word clouds
EMOTION_COLORS = {
    'joy': 'YlOrBr',    # Yellow-Orange-Brown for joy
    'anger': 'Reds',    # Red for anger
    'disgust': 'Greens',  # Green for disgust
    'sadness': 'Blues',   # Blue for sadness
    'fear': 'Purples',    # Purple for fear
    'surprise': 'Oranges' # Orange for surprise
}

def generate_word_clouds_by_emotion(df_movies_with_emotions, top_emotions=1):
    """
    Generates and displays word clouds based on the most frequent words for each emotion.

    Args:
    df_movies_with_emotions (pd.DataFrame): DataFrame containing movie plot texts and emotion scores.
    top_emotions (int): Number of top emotions to consider per plot.
    """
    # Columns for normalized emotion scores
    emotion_columns = [f"normalized_plot_{emotion}_without_neutral" for emotion in EMOTIONS]

    # Identify the top emotions for each movie and store them in a new column
    df_movies_with_emotions["top_emotions"] = df_movies_with_emotions[
        emotion_columns
    ].apply(lambda row: row.nlargest(top_emotions).index.tolist(), axis=1)

    # Initialize lemmatizer and stop words
    lemmatizer = WordNetLemmatizer()
    stop_words = set(stopwords.words("english")).union(ENGLISH_STOP_WORDS)

    # Initialize word count counters for each emotion
    word_counts_by_emotion = {emotion: Counter() for emotion in emotion_columns}

    # Process each movie plot and update word counts for associated emotions
    for _, row in df_movies_with_emotions.iterrows():
        plot_text = row["plot"].lower()
        for emotion in row["top_emotions"]:
            if emotion in word_counts_by_emotion:
                # Tokenize, clean, and lemmatize words while removing stop words
                words = [
                    lemmatizer.lemmatize(word)
                    for word in re.findall(r"\b\w+\b", plot_text)
                    if word not in stop_words and lemmatizer.lemmatize(word) != "tell"
                ]
                word_counts_by_emotion[emotion].update(words)

    # Ensure the output directory exists
    os.makedirs(PLOTS_PATH, exist_ok=True)

    # Generate and save word clouds for each emotion
    for emotion, word_counts in word_counts_by_emotion.items():
        if word_counts:
            emotion_name = emotion.split('_')[2]  # Extract emotion name

            # Load the image mask for the emotion
            mask_img = Image.open(f'./data/images/{emotion_name}.png').convert("RGBA")
            mask_img = resize_image(mask_img, factor=2)

            # Create a white background image for the word cloud
            output_img = Image.new("RGBA", mask_img.size, (255, 255, 255, 255))
            output_img = apply_mask_to_image(mask_img, output_img)

            # Generate a NumPy array for the word cloud mask
            black_mask = np.array(output_img.convert("RGB"))

            # Create and generate the word cloud
            wordcloud = WordCloud(
                width=800,
                height=400,
                background_color="white",
                colormap=EMOTION_COLORS.get(emotion_name, 'viridis'),
                max_words=100,
                mask=black_mask,
            ).generate_from_frequencies(word_counts)

            # Display the word cloud
            plt.figure(figsize=(10, 5))
            plt.imshow(wordcloud, interpolation="bilinear")
            plt.axis("off")
            plt.show()

            # Save the word cloud image
            save_wordcloud_image(wordcloud, emotion_name)

            # Create a personalized version with reduced opacity for the mask
            personalized_img = create_personalized_mask(emotion_name)
            combined_image = Image.alpha_composite(wordcloud.to_image().convert("RGBA"), personalized_img)

            # Save the combined image
            save_combined_image(combined_image, emotion_name)

    # Afficher les 10 mots les plus fréquents pour chaque émotion
    print("\nTop 10 mots par émotion:")
    for emotion, word_counts in word_counts_by_emotion.items():
        if word_counts:
            emotion_name = emotion.split('_')[2]  # Extraire le nom de l'émotion
            top_words = word_counts.most_common(10)
            print(f"\nÉmotion: {emotion_name.capitalize()}")
            for word, count in top_words:
                print(f"  {word}: {count}")



def resize_image(image, factor=2):
    """
    Resizes an image by a given factor.

    Args:
    image (PIL.Image): Image to resize.
    factor (int): Resize factor.
    Returns:
    PIL.Image: Resized image.
    """
    return image.resize((factor * image.width, factor * image.height), Image.Resampling.LANCZOS)


def apply_mask_to_image(input_img, output_img):
    """
    Applies a mask to an image, turning visible pixels to black and transparent ones to white.

    Args:
    input_img (PIL.Image): The input image with a mask.
    output_img (PIL.Image): The output image to apply the mask to.

    Returns:
    PIL.Image: Image with mask applied.
    """
    input_pixels = input_img.load()
    output_pixels = output_img.load()

    for y in range(input_img.height):
        for x in range(input_img.width):
            r, g, b, a = input_pixels[x, y]
            if a > 0:  # If pixel is visible (opaque or semi-transparent)
                output_pixels[x, y] = (0, 0, 0, 255)  # Set pixel to opaque black
            else:  # If pixel is transparent
                output_pixels[x, y] = (255, 255, 255, 255)  # Set pixel to opaque white

    return output_img


def save_wordcloud_image(wordcloud, emotion_name):
    """
    Saves the generated word cloud image.

    Args:
    wordcloud (WordCloud): The word cloud object.
    emotion_name (str): The emotion name used for file naming.
    """
    filename = f"{emotion_name.capitalize()}_Word_Cloud.png"
    filepath = os.path.join(PLOTS_PATH, filename)
    wordcloud.to_file(filepath)


def create_personalized_mask(emotion_name):
    """
    Creates a personalized mask with reduced opacity for the emotion's image.

    Args:
    emotion_name (str): The emotion name for the image.

    Returns:
    PIL.Image: Personalized mask image.
    """
    img = Image.open(f'./data/images/{emotion_name}.png').convert("RGBA")
    img = resize_image(img, factor=2)

    for x in range(img.width):
        for y in range(img.height):
            r, g, b, a = img.getpixel((x, y))
            new_a = int(a * 0.1)  # Reduce opacity by 90%
            img.putpixel((x, y), (r, g, b, new_a))

    return img


def save_combined_image(combined_image, emotion_name):
    """
    Saves the combined image of the word cloud and the personalized mask.

    Args:
    combined_image (PIL.Image): The combined image to save.
    emotion_name (str): The emotion name for file naming.
    """
    filename = f"{emotion_name.capitalize()}_Word_Cloud_Personified.png"
    filepath = os.path.join(PLOTS_PATH, filename)
    combined_image.save(filepath)










def show_top_10_words_per_emotion(df_movies_with_emotions_normalized, top_emotions=1):
    """
    Displays and saves bar plots of the top 10 words associated with each dominant emotion in movie plots.

    Args:
    df_movies_with_emotions (pd.DataFrame): DataFrame containing columns for normalized emotion scores and plots.
    top_emotions (int): Number of top emotions to consider per plot.
    output_dir (str): Directory path where plots will be saved.
    """

    # Define the emotions to analyze
    emotion_columns = [f"normalized_plot_{emotion}_without_neutral" for emotion in EMOTIONS]

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
    os.makedirs(PLOTS_PATH, exist_ok=True)

    # Generate bar plots for the top 10 words in each emotion category using Plotly
    for idx, (emotion, word_counts) in enumerate(word_counts_by_emotion.items()):
        common_words = word_counts.most_common(10)
        if common_words:
            words, counts = zip(*common_words)
            emotion_name = emotion.split('_')[2]

            # Create a bar chart using Plotly
            fig = go.Figure(
                data=[
                    go.Bar(
                        x=words, 
                        y=counts, 
                        text=counts,
                        marker_color=COLORS.get(emotion_name, '#000000'),
                        hovertemplate=(
                            "<b>Emotion:</b> " + emotion_name + "<br>"
                            "<b>Word:</b> %{x}<br>"
                            "<b>Count:</b> %{y}<extra></extra>"
                        ),
                    )
                ]
            )

            # Update the layout for better styling
            fig.update_layout(
                title=f"Top 10 Words in {emotion_name.capitalize()} Movie Plots",
                xaxis_title="Words",
                yaxis_title="Counts",
                template="plotly_white",
                showlegend=False,
                title_x=0.5,
                title_xanchor="center"
            )

            # Show the plot
            fig.show()

            # Save the figure
            filename = f"{emotion_name.capitalize()}_Top_10_Words.html"
            filepath = os.path.join(PLOTS_PATH, f"{filename}.html")
            fig.write_html(filepath)