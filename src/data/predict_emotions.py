import pandas as pd
from transformers import pipeline
import json
from tqdm import tqdm
import re

MAX_CHAR = 1000 # maximum character per sentence for model compatibility
EMOTIONS = ['anger', 'disgust', 'fear', 'joy', 'neutral', 'sadness', 'surprise'] # list of supported emotions by the model

# Initialize the classifier
classifier = pipeline("text-classification", model="j-hartmann/emotion-english-distilroberta-base", top_k=None)


def read_tsv(file_path):
    '''
    Reads a TSV file and returns a DataFrame, or None if the file doesn't exist.
    
    Args:
        file_path (str): TSV file path.

    Returns:
        DataFrame: DataFrame corresponding to the TSV file.
    '''
    try:
        return pd.read_csv(file_path, sep='\t')
    except FileNotFoundError:
        return None


def write_tsv(df, file_path, mode='w'):
    '''
    Writes a DataFrame to a TSV file.
    
    Args:
        df (DataFrame): DataFrame to be saved.
        file_path (str): TSV file path.
        
    Returns:
        None
    '''
    df.to_csv(file_path, sep='\t', index=False, mode=mode)


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
    
    for sentence in sentences:
        results = classifier(sentence)
        predictions = {emotion: 0 for emotion in scores.keys()}
        
        for result in results[0]:
            predictions[result['label']] += result['score']
        
        for emotion in scores:
            scores[emotion].append(predictions[emotion])

    return {'sentences': sentences, **scores}


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
    num_sentences = len(sentences)
    
    for sentence in sentences:
        results = classifier(sentence)
        
        for result in results[0]:
            scores[result['label']] += result['score']
    
    # Normalize scores if there are sentences
    if num_sentences > 0:
        total_score = sum(scores.values())
        scores = {emotion: score / total_score for emotion, score in scores.items()}

    return {'sentences': sentences, **scores}


def predict_emotions_to_tsv(df, column, file_path, id_column='wikipedia_ID', is_review=False):
    '''
    Predicts emotions for each entry in the specified column of the DataFrame and saves results to a TSV file.
    Handles cases where multiple rows may share the same ID by using row indexes.

    Args:
        df (pd.DataFrame): DataFrame containing text data for prediction.
        column (str): Name of the column containing text to analyze.
        file_path (str): Path to the TSV file where results will be saved.
        id_column (str): Name of the column containing the unique ID for identifying rows. Default is 'wikipedia_ID'.
        is_review (bool): Flag indicating whether the text is a review (True) or plot (False). Default is False.

    Returns:
        None
    '''
    existing_data = read_tsv(file_path)

    # Find the last processed row index if the TSV file exists
    last_index = None
    if existing_data is not None:
        last_index = len(existing_data) - 1  # Last written row

    start_writing = last_index is None  # Start from beginning if no previous data

    # Open TSV for appending new data
    with open(file_path, 'a') as file:
        if start_writing:
            file.write(f"{id_column}\temotion_predictions\n")

        for idx, row in tqdm(df.iterrows(), total=len(df), desc="Processing and saving"):
            
            if last_index is not None and idx <= last_index:  # Skip rows up to the last processed index
                continue

            # Predict emotions for the current row's text column
            if is_review:
                emotion_prediction = calculate_review_emotion_scores(row[column])
            else:
                emotion_prediction = calculate_plot_emotion_scores(row[column])
            
            emotion_json = json.dumps(emotion_prediction)

            # Append row to TSV
            file.write(f"{row[id_column]}\t{emotion_json}\n")





def merge_df_with_emotions_tsv(df, file_path, prefix, is_review=False):
    '''
    Merges emotion predictions from a TSV into a DataFrame, with prefixed column names.
    If is_review is True, merge in order. Otherwise, merge by 'wikipedia_ID'.

    Args:
        df (pd.DataFrame): DataFrame to merge with predictions.
        file_path (str): Path to the TSV file with predictions.
        prefix (str): Prefix for new columns.
        is_review (bool): If True, merge in order of rows; otherwise, merge by 'wikipedia_ID'.

    Returns:
        pd.DataFrame: Merged DataFrame with original data and emotion scores.
    '''
    df_emotions = read_tsv(file_path)
    if df_emotions is None:
        raise FileNotFoundError(f"No TSV file found at '{file_path}' to merge.")
    
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
        merged_df = pd.concat([df, df_emotions], axis=1)
    else:
        # If not reviews, merge by 'wikipedia_ID'
        merged_df = pd.merge(df, df_emotions, on='wikipedia_ID', how='left')

    return merged_df

