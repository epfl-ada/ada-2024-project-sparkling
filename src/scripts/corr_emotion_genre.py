import numpy as np 
import matplotlib.pyplot as plt
import pandas as np
import seaborn as sns

NBR_GENRES = 10 # to change

def vector_genres(idx, size=NBR_GENRES):
    vect = np.zeros(size)
    vect[idx] = 1
    return vect

def correlation_genres_emotions(df, emotions, genres):
    '''
    Plots a representation of the correlation between each pair of emotion-genre.
    '''
    df['emotions'] = df[emotions].copy().apply(lambda x: x.tolist(), axis=1)
    df['genres'] = df.genre.copy().apply(lambda x: vector_genres(idx, NBR_GENRES))
    
    df_combined = pd.concat([df.emotions, df.genres], axis=1)
    
    correlation_mtx = df_combined.corr().loc['emotions', 'genres']
    
    plt.figure()
    sns.heatmap(correlation_mtx, annot=True, center=0)
    plt.title('Correlation between genres and emotions')
    plt.show()
    return None
    