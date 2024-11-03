import pandas as pd
from transformers import pipeline
from tqdm import tqdm
import re

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
