import pandas as pd
from transformers import pipeline
from tqdm import tqdm

# Predicts the emotions present in a given movie plot by using a pre-trained classifier.
def predict_emotions_from_plot(classifier, plot: str):
    predictions = {
        'anger': 0,
        'disgust': 0,
        'fear': 0,
        'joy': 0,
        'neutral': 0,
        'sadness': 0,
        'surprise': 0
    }
    
    sentences = plot.split('.')
    if sentences[-1].strip() == "":
        sentences = sentences[:-1]
    
    for s in sentences:
        results = classifier(s.strip())[0]
        
        for result in results:
            label = result['label']
            score = result['score']
            predictions[label] += score
            
    total_score = sum(predictions.values())
    normalized_predictions = {label: score / total_score for label, score in predictions.items()}
    
    return normalized_predictions

# Adds emotion prediction columns to a DataFrame of movies based on their plots.
def add_emotions_to_df(df_movies):
    classifier = pipeline("text-classification", model="j-hartmann/emotion-english-distilroberta-base", top_k=None)
    tqdm.pandas(desc="Processing emotions")
    emotion_predictions = df_movies['plot'].progress_apply(lambda plot: predict_emotions_from_plot(classifier, plot))
    df_emotion = pd.DataFrame(emotion_predictions.tolist())
    df_movies_with_emotions = pd.concat([df_movies, df_emotion], axis=1)
    
    return df_movies_with_emotions
