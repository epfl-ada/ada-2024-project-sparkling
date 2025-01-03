import pandas as pd
from transformers import pipeline
import json
from tqdm import tqdm
import re
import os

from src.data.load_data import DATASETS_DIR

MAX_CHAR = 1000  # Maximum character per sentence for model compatibility
EMOTIONS = ['anger', 'disgust', 'fear', 'joy', 'neutral', 'sadness', 'surprise']  # List of supported emotions by the model

# Initialize the pre-trained classifier
classifier = pipeline("text-classification", model="j-hartmann/emotion-english-distilroberta-base", top_k=None)

# Define the data directory
PREDICTED_EMOTIONS_DIR = os.path.join(DATASETS_DIR, 'PredictedEmotions')

# Ensure the data directory exists
os.makedirs(PREDICTED_EMOTIONS_DIR, exist_ok=True)

def read_tsv_predicted_emotions(file_name):
    '''
    Reads a TSV file from the src/data directory and returns a DataFrame, or None if the file doesn't exist.
    
    Args:
        file_name (str): Name of the TSV file (should be in src/data folder).

    Returns:
        DataFrame: DataFrame corresponding to the TSV file.
    '''
    file_path = os.path.join(PREDICTED_EMOTIONS_DIR, file_name)  # Construct file path
    try:
        return pd.read_csv(file_path, sep='\t')
    except FileNotFoundError:
        return None


def write_tsv(df, file_name, mode='w', header=True):
    '''
    Writes a DataFrame to a TSV file in the src/data directory.
    
    Args:
        df (DataFrame): DataFrame to be saved.
        file_name (str): Name of the TSV file (should be in src/data folder).
        
    Returns:
        None
    '''
    file_path = os.path.join(PREDICTED_EMOTIONS_DIR, file_name)  # Construct file path
    df.to_csv(file_path, sep='\t', index=False, mode=mode, header=header)


def preprocess_text(text):
    '''
    Preprocesses text by removing HTML tags, replacing specific characters, and splitting into truncated sentences.
    
    Args:
        text (str): Text to preprocess before analyzing for emotions.

    Returns:
        list: List of sentences.
    '''
    text = re.sub(r'<.*?>', '', text).replace('&ndash;', '–').strip()
    sentences = text.split('.')
    
    if sentences and sentences[-1].strip() == "":
        sentences = sentences[:-1]
    
    final_sentences = []
    for s in sentences:
        while len(s) > MAX_CHAR:
            final_sentences.append(s[:MAX_CHAR].strip())
            s = s[MAX_CHAR:]
        final_sentences.append(s.strip())
    
    return final_sentences


def calculate_plot_emotion_scores(text):
    '''
    Calculates emotion scores per sentence for a plot.

    Args:
        text (str): Plot text to analyze for emotions.

    Returns:
        dict: Dictionary with a list of sentences and individual emotion scores per sentence.
    '''
    sentences = preprocess_text(text)
    scores = {emotion: [] for emotion in EMOTIONS}
    valid_sentences = []

    for idx, sentence in enumerate(sentences):
        try:
            results = classifier(sentence)
        except Exception as e:
            print(f"Error processing sentence {sentence} at index {idx} from text {text}: {e}")
            continue
        
        predictions = {emotion: 0 for emotion in scores.keys()}
        
        for result in results[0]:
            predictions[result['label']] += result['score']
        
        valid_sentences.append(sentence)
        
        for emotion in scores:
            scores[emotion].append(predictions[emotion])

    return {'sentences': valid_sentences, **scores}


def calculate_review_emotion_scores(text):
    '''
    Calculates normalized emotion scores across the entire review.

    Args:
        text (str): Review text to analyze for emotions.

    Returns:
        dict: Dictionary with normalized emotion scores across the text.
    '''
    sentences = preprocess_text(text)
    scores = {emotion: 0 for emotion in EMOTIONS}
    valid_sentences = []
    
    for idx, sentence in enumerate(sentences):
        try:
            results = classifier(sentence)
        except Exception as e:
            print(f"Error processing sentence {sentence} at index {idx} from text {text}: {e}")
            continue
            
        valid_sentences.append(sentence)
        
        for result in results[0]:
            scores[result['label']] += result['score']
    
    # Normalize scores if there are valid sentences
    num_sentences = len(valid_sentences)
    if num_sentences > 0:
        total_score = sum(scores.values())
        if total_score > 0:  # Prevent division by zero
            scores = {emotion: score / total_score for emotion, score in scores.items()}
    
    return {'sentences': valid_sentences, **scores}


def predict_emotions_to_tsv(df, column, file_name, id_column='wikipedia_ID', is_review=False):
    '''
    Predicts emotions for each entry in the specified column of the DataFrame and saves results to a TSV file.
    Resumes from the last processed row.

    Args:
        df (pd.DataFrame): DataFrame containing text data for prediction.
        column (str): Name of the column containing text to analyze.
        file_name (str): Name of the TSV file to save the predictions (should be in src/data folder).
        id_column (str): Name of the column containing the unique ID for identifying rows. Default is 'wikipedia_ID'.
        is_review (bool): Flag indicating whether the text is a review (True) or plot (False). Default is False.

    Returns:
        None
    '''
    
    # Read existing TSV to find how many rows are processed
    existing_data = read_tsv_predicted_emotions(file_name)  # Check if TSV file exists

    if existing_data is None:
        # Initialize TSV with column names if file doesn't exist
        write_tsv(pd.DataFrame(columns=[id_column, 'emotion_predictions']), file_name, header=True)
        start_idx = 0  # Start from the beginning if no existing data
    else:
        # Determine the start index based on the number of rows in the existing file
        start_idx = len(existing_data)  # Resume from the last processed row

    # Loop through the DataFrame starting from the last unprocessed row
    for idx, row in tqdm(df.iloc[start_idx:].iterrows(), total=len(df) - start_idx, desc="Processing and saving"):
        # Predict emotions for the current row's text column
        if is_review:
            emotion_prediction = calculate_review_emotion_scores(row[column])
        else:
            emotion_prediction = calculate_plot_emotion_scores(row[column])
        
        emotion_json = json.dumps(emotion_prediction)  # Serialize the prediction to JSON format
        
        # Prepare the row to append
        row_to_write = {id_column: row[id_column], 'emotion_predictions': emotion_json}
        
        # Save the current row to the TSV file immediately
        write_tsv(pd.DataFrame([row_to_write]), file_name, mode='a', header=False)




def merge_df_with_emotions_tsv(df, file_name, prefix, is_review=False):
    '''
    Merges emotion predictions from a TSV into a DataFrame, with prefixed column names.
    If is_review is True, merge in order. Otherwise, merge by 'wikipedia_ID'.

    Args:
        df (pd.DataFrame): DataFrame to merge with predictions.
        file_name (str): Name of the TSV file to merge predictions from (should be in src/data folder).
        prefix (str): Prefix for new columns.
        is_review (bool): If True, merge in order of rows; otherwise, merge by 'wikipedia_ID'.

    Returns:
        pd.DataFrame: Merged DataFrame with original data and emotion scores.
    '''
    df_emotions = read_tsv_predicted_emotions(file_name)
    if df_emotions is None:
        raise FileNotFoundError(f"No TSV file found at '{file_name}' to merge.")
    
    # Parse the emotion predictions from JSON
    df_emotions['emotion_predictions'] = df_emotions['emotion_predictions'].apply(json.loads)
    
    # Extract individual emotion scores and create new columns with the specified prefix
    for key in df_emotions['emotion_predictions'].iloc[0].keys():
        df_emotions[f"{prefix}_{key}"] = df_emotions['emotion_predictions'].apply(lambda x: x[key])
    
    # Drop the 'emotion_predictions' column after expanding it
    df_emotions.drop(columns=['emotion_predictions'], inplace=True)

    # Merge in order or by 'wikipedia_ID' based on the 'is_review' argument
    if is_review:
        # If reviews, merge in order of rows (i.e., keep the rows aligned)
        merged_df = pd.concat([df, df_emotions.drop(['wikipedia_ID'], axis=1)], axis=1)
    else:
        # If not reviews, merge by 'wikipedia_ID'
        merged_df = pd.merge(df, df_emotions, on='wikipedia_ID', how='left')

    return merged_df
