import pandas as pd
from transformers import pipeline
import json
from tqdm import tqdm
import re
import os


# Predicts the emotions present in a given movie plot by using a pre-trained classifier.
def predict_emotions_from_plot(classifier, plot: str):
    plot = plot.strip()
    plot = re.sub(r'<.*?>', '', plot)
    plot = plot.replace('&ndash;', '–')
    
    # Return None for each emotion if the plot is empty
    if plot.strip() == "":
        return {label: None for label in predictions}
    
    # Split the plot into sentences
    sentences = plot.split('.')
    if sentences[-1].strip() == "": # Remove the last element if empty
        sentences = sentences[:-1]
        
    # Sentences fed to the model
    final_sentences = []
    
    # Split sentences that are too long for the model
    for s in sentences:
        while len(s) > 1300:
            final_sentences.append(s[:1300].strip())
            s = s[1300:]
    
        final_sentences.append(s.strip())
        
    anger_scores = []
    disgust_scores = []
    fear_scores = []
    joy_scores = []
    neutral_scores = []
    sadness_scores = []
    surprise_scores = []
        
    for s in final_sentences:
        results = classifier(s) # classify the sentence into the 7 emotions
        if not results:
            continue
            
        results = results[0]
        
        # initializing emotion scores for the current sentence
        predictions = {
            'anger': 0,
            'disgust': 0,
            'fear': 0,
            'joy': 0,
            'neutral': 0,
            'sadness': 0,
            'surprise': 0
        }
        
        # updating emotion scores for the current sentence
        for result in results:
            label = result['label']
            score = result['score']
            predictions[label] += score
            
        # adding each emotion score to their scores list for the plot
        anger_scores.append(predictions['anger'])
        disgust_scores.append(predictions['disgust'])
        fear_scores.append(predictions['fear'])
        joy_scores.append(predictions['joy'])
        neutral_scores.append(predictions['neutral'])
        sadness_scores.append(predictions['sadness'])
        surprise_scores.append(predictions['surprise'])
        
    # return a JSON to be able to retrieve list objects when merging
    emotion_predictions = {
        'plot_sentences': final_sentences,
        'anger': anger_scores,
        'disgust': disgust_scores,
        'fear': fear_scores,
        'joy': joy_scores,
        'neutral': neutral_scores,
        'sadness': sadness_scores,
        'surprise': surprise_scores
    }
        
    return emotion_predictions



# Adds emotion prediction columns to a DataFrame of movies based on their plots and save them into a csv.
def predict_emotions_to_csv(df_movies, column_to_predict, csv_path):
    classifier = pipeline("text-classification", model="j-hartmann/emotion-english-distilroberta-base", top_k=None)
    tqdm.pandas(desc="Processing emotions")
    
    # Emotion predictions
    if column_to_predict == 'plot':
        emotion_predictions = df_movies[column_to_predict].progress_apply(lambda plot: predict_emotions_from_plot(classifier, plot))
    elif column_to_predict == 'review_detail':
        emotion_predictions = df_movies[column_to_predict].progress_apply(lambda review: predict_emotions_from_review(classifier, review))
    else:
        raise ValueError("Argument 'column_to_predict' should be 'plot' or 'review_detail'.") 
    
    # Convert predictions to JSON
    df_movies['emotion_predictions'] = emotion_predictions.apply(lambda x: json.dumps(x))
    
    # Save the DataFrame with columns  wikipedia_ID and emotion_predictions
    df_movies.to_csv(csv_path, 
                     columns=['wikipedia_ID', 'emotion_predictions'], 
                     index=False)
    
    
    
# Function to merge the emotion predictions CSV with df_movies on wikipedia_ID
def merge_df_movies_with_emotions_csv(df_movies, csv_path):
    # Read CSV with emotion predictions
    df_emotions = pd.read_csv(csv_path)
    
    # Decode column 'emotion_predictions' from JSON
    df_emotions['emotion_predictions'] = df_emotions['emotion_predictions'].apply(lambda x: json.loads(x))
    
    # Extract data from predictions into separate columns
    df_emotions['plot_sentences'] = df_emotions['emotion_predictions'].apply(lambda x: x['plot_sentences'])
    df_emotions['anger'] = df_emotions['emotion_predictions'].apply(lambda x: x['anger'])
    df_emotions['disgust'] = df_emotions['emotion_predictions'].apply(lambda x: x['disgust'])
    df_emotions['fear'] = df_emotions['emotion_predictions'].apply(lambda x: x['fear'])
    df_emotions['joy'] = df_emotions['emotion_predictions'].apply(lambda x: x['joy'])
    df_emotions['neutral'] = df_emotions['emotion_predictions'].apply(lambda x: x['neutral'])
    df_emotions['sadness'] = df_emotions['emotion_predictions'].apply(lambda x: x['sadness'])
    df_emotions['surprise'] = df_emotions['emotion_predictions'].apply(lambda x: x['surprise'])
    
    # Merge with main DataFrame
    merged_df = pd.merge(df_movies, df_emotions, on='wikipedia_ID')
    merged_df = merged_df.drop('emotion_predictions', axis=1)
    return merged_df



# Predicts the emotions present in a given movie plot by using a pre-trained classifier.
def predict_emotions_from_review(classifier, review: str):
    review = review.strip()
    review = re.sub(r'<.*?>', '', review)
    review = review.replace('&ndash;', '–')
    
    # initializing emotion scores for the review
    predictions = {
        'anger': 0,
        'disgust': 0,
        'fear': 0,
        'joy': 0,
        'neutral': 0,
        'sadness': 0,
        'surprise': 0
    }
    
    # Return None for each emotion if the plot is empty
    if review.strip() == "":
        return {label: None for label in predictions}
        
    # Sentences fed to the model
    sentences = []
    
    # Split into different sentences if length of review is larger than 1300
    for i in range(0, len(review), 1300):
        sentences.append(review[i:i+1300])
    
    num_sentences = len(sentences)
        
    for s in sentences:
        results = classifier(s) # classify the sentence into the 7 emotions
        if not results:
            num_sentences -= 1
            continue
            
        results = results[0]
        
        # updating emotion scores for the current sentence
        for result in results:
            label = result['label']
            score = result['score']
            predictions[label] += score
            
    # Average the scores across all sentences
    if num_sentences > 0:
        predictions = {label: score / num_sentences for label, score in predictions.items()}
        
    # Return a JSON to be able to retrieve list objects when merging
    emotion_predictions = {
        'review_sentences': sentences,
        'review_anger': predictions['anger'],
        'review_disgust': predictions['disgust'],
        'review_fear': predictions['fear'],
        'review_joy': predictions['joy'],
        'review_neutral': predictions['neutral'],
        'review_sadness': predictions['sadness'],
        'review_surprise': predictions['surprise']
    }
        
    return emotion_predictions


# Function to merge the emotion predictions CSV on reviews with df_movies on wikipedia_ID
def merge_df_reviews_with_emotions_csv(df_movies, csv_path):
    # Read CSV with emotion predictions
    df_emotions = pd.read_csv(csv_path)
    
    # Decode column 'emotion_predictions' from JSON
    df_emotions['emotion_predictions'] = df_emotions['emotion_predictions'].apply(lambda x: json.loads(x))
    
    # Extract data from predictions into separate columns
    df_emotions['review_sentences'] = df_emotions['emotion_predictions'].apply(lambda x: x['review_sentences'])
    df_emotions['review_anger'] = df_emotions['emotion_predictions'].apply(lambda x: x['review_anger'])
    df_emotions['review_disgust'] = df_emotions['emotion_predictions'].apply(lambda x: x['review_disgust'])
    df_emotions['review_fear'] = df_emotions['emotion_predictions'].apply(lambda x: x['review_fear'])
    df_emotions['review_joy'] = df_emotions['emotion_predictions'].apply(lambda x: x['review_joy'])
    df_emotions['review_neutral'] = df_emotions['emotion_predictions'].apply(lambda x: x['review_neutral'])
    df_emotions['review_sadness'] = df_emotions['emotion_predictions'].apply(lambda x: x['review_sadness'])
    df_emotions['review_surprise'] = df_emotions['emotion_predictions'].apply(lambda x: x['review_surprise'])
    
    # Merge with main DataFrame
    merged_df = pd.merge(df_movies, df_emotions, on='wikipedia_ID')
    merged_df = merged_df.drop('emotion_predictions', axis=1)
    return merged_df