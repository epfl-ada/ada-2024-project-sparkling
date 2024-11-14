import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

def generate_emotion_genre_heatmap(df_genres, df_emotions):
    
    genre_columns = df_genres.columns.difference(['wikipedia_ID'])
    emotion_columns = ['normalized_plot_anger', 'normalized_plot_disgust', 'normalized_plot_fear', 
                       'normalized_plot_joy', 'normalized_plot_sadness', 'normalized_plot_surprise']

    #To have in each line the emotions and genres corresponding to the same movie
    df_with_emotions_and_genres = pd.merge(df_genres, df_emotions, on='wikipedia_ID', how='inner')

    #Calcultion avec the correlation matrix
    correlation_matrix = pd.concat([df_with_emotions_and_genres[emotion_columns], 
                                    df_with_emotions_and_genres[genre_columns].astype(float)], 
                                    axis=1).corr(method='pearson').loc[emotion_columns, genre_columns] 

    #Plotting the data 
    emotion_labels = [emotion.replace('normalized_plot_', '') for emotion in emotion_columns]
    plt.figure(figsize=(12, 8))
    sns.heatmap(correlation_matrix, cmap="coolwarm", center=0, cbar=True, annot=True, fmt=".2f",
                annot_kws={"size": 7}, linewidths=0.5, yticklabels=emotion_labels)
    plt.xticks(rotation=45, ha='right', fontsize=9)
    plt.yticks(rotation=0, fontsize=9)
    plt.title('Correlation between emotions and genres', fontsize=14)
    plt.tight_layout()
    plt.show()

def plot_top_country_emotion_correlation(df_emotions, df_country):
    
    emotion_columns = ['normalized_plot_anger', 'normalized_plot_disgust', 'normalized_plot_fear', 
                       'normalized_plot_joy', 'normalized_plot_sadness', 'normalized_plot_surprise']
    
    #To have in each line the emotions and country corresponding to the same movie
    df_with_emotions_and_contries = pd.merge(df_emotions, df_country, on='wikipedia_ID', how='inner')

    #Make a list of the different countries
    country_columns = df_country.columns.difference(['wikipedia_ID'])

    #Calcultion avec the correlation matrix
    correlation_matrix = pd.concat([df_with_emotions_and_contries[emotion_columns], 
                                    df_with_emotions_and_contries[country_columns]], 
                                    axis=1).corr(method='pearson').loc[emotion_columns, country_columns]

    #Create a graph for each emotion
    for emotion in emotion_columns:
        #Obtain the top 10 countries with the strongest positive and negative correlations for each emotion
        top_positive = correlation_matrix.loc[emotion].nlargest(10)
        top_negative = correlation_matrix.loc[emotion].nsmallest(10)
        top_countries = pd.concat([top_positive, top_negative])

        plt.figure(figsize=(12, 8))
        sns.barplot(x=top_countries.index, y=top_countries.values, palette="coolwarm", legend=False, hue=top_countries.index)
        plt.title(f"Top 10 positive and negative country correlations with {emotion.replace('normalized_plot_', '')}")
        plt.xlabel("Country")
        plt.ylabel("Correlation")
        plt.xticks(rotation=45, ha="right")
        plt.tight_layout()
        plt.show()

def plot_emotion_distribution_per_period(dataframe, period_column):
    
    emotions = ['normalized_plot_anger', 'normalized_plot_disgust', 'normalized_plot_fear', 
                'normalized_plot_joy', 'normalized_plot_sadness', 'normalized_plot_surprise']
    
    # calculate the proportion of each emotion for each period
    period_emotion_totals = dataframe.groupby(period_column)[emotions].sum()
    period_emotion_distribution = period_emotion_totals.div(period_emotion_totals.sum(axis=1), axis=0)
    
    # Create the plot
    plt.figure(figsize=(14, 8))
    bottom = None
    colors = sns.color_palette("Dark2", len(emotions))  # Palette Spectral pour chaque émotion

    #One bar for each emotion
    for i, emotion in enumerate(emotions):
        emotion_label = emotion.replace("normalized_plot_", "")
        if bottom is None:
            bottom = period_emotion_distribution[emotion]
            plt.bar(period_emotion_distribution.index, period_emotion_distribution[emotion], 
                    label=emotion_label, color=colors[i])
        else:
            plt.bar(period_emotion_distribution.index, period_emotion_distribution[emotion], bottom=bottom, 
                    label=emotion_label, color=colors[i])
            bottom += period_emotion_distribution[emotion]

    plt.title(f"Distribution of emotions per {period_column.replace('release_', '')}")
    plt.xlabel(period_column.replace('release_', '').capitalize())
    plt.ylabel("Proportion")
    plt.legend(title="Emotion", bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.show()

def calculate_deviation(dataframe, reference_period='release_month'):
    
    emotions = ['normalized_plot_anger', 'normalized_plot_disgust', 'normalized_plot_fear', 
                'normalized_plot_joy', 'normalized_plot_sadness', 'normalized_plot_surprise']
    
    overall_mean = dataframe[emotions].mean()
    period_data = dataframe.groupby(reference_period)[emotions].mean()

    deviation_index = period_data - overall_mean

    #Plotting deviations
    plt.figure(figsize=(12, 6))
    for emotion in emotions:
        emotion_label = emotion.replace("normalized_plot_", "")
        plt.plot(deviation_index.index, deviation_index[emotion], label=emotion_label)
        
    plt.title("Index of deviation of emotions from the mean")
    plt.xlabel("Month")
    plt.ylabel("Deviation from the mean")
    plt.legend(title="Emotion", bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.show()

def plot_major_emotion_per_five_years(dataframe, year_column='release_year'):
    
    emotions = ['normalized_plot_anger', 'normalized_plot_disgust', 'normalized_plot_fear', 
                'normalized_plot_joy', 'normalized_plot_sadness', 'normalized_plot_surprise']
    
    #Group the data by five year intervals, using the beginning year of each period
    dataframe['period'] = (dataframe[year_column] // 5) * 5
    
    #Calculate the mean emotion scores for each period
    period_means = dataframe.groupby('period')[emotions].mean()
    
    #Find the dominant emotion and its percentage for each period
    dominant_emotion = period_means.idxmax(axis=1).str.replace('normalized_plot_', '')
    dominant_percentage = period_means.max(axis=1)
    
    #Add dominant emotion and percentage as new columns to period_means
    period_means['Dominant Emotion'] = dominant_emotion
    period_means['Percentage'] = dominant_percentage
    
    #Plotting
    plt.figure(figsize=(14, 6))
    sns.barplot(x=period_means.index, y='Percentage', hue='Dominant Emotion', data=period_means)
    plt.title("Dominant emotion for each quinquennium")
    plt.xlabel("Five year period")
    plt.ylabel("Proportion")
    plt.legend(title="Emotion", bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.tight_layout()
    plt.show()