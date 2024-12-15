def get_actors_name_and_statistics(df_actor_id, df_characters, df_movies_with_normalized_emotions):
    """
    Given a dataframe containing actors ids, create a copy and extends this data frame and add additional statistics and informations about the actor.
    First, this function will add the name of the actors. Then it will add the number of movies contained in our data set along with for each emotions
    the number of movies in which it is the dominant emotion.

    Arguments:
        - df_actor_id: Dataframe containing a column "actor_id" containing the freebase actor id of the actors
        - df_characters: Dataframe containing the actors and character information associated with the movie they played in 
        - df_movies_with_normalized_emotions: Dataframe containing the emotions of the plots 
    """
    df_actor_id = df_actor_id.copy()

    # Get actor name
    df_actor_id["actor_name"] = df_actor_id["actor_id"].apply(lambda actor_id: df_characters[df_characters["freebase_ID_actor"] == actor_id]["actor_name"].iloc[0])

    # Get actor movies count
    df_characters = df_characters.drop_duplicates()
    df_character_filtered = df_characters[df_characters["wikipedia_ID"].isin(df_movies_with_normalized_emotions["wikipedia_ID"])]
    df_character_filtered = df_character_filtered[df_character_filtered["freebase_ID_actor"].isin(df_actor_id["actor_id"])]
    df_actors_count = df_character_filtered.groupby("freebase_ID_actor")["wikipedia_ID"].count()

    df_actor_id["actor_movies_count"] = df_actor_id["actor_id"].apply(lambda actor_id: df_actors_count[df_actors_count.index == actor_id].iloc[0])

    #Get actor movies count emotions
    df_characters_movies = df_characters.merge(df_movies_with_normalized_emotions, on="wikipedia_ID")
    df_characters_movies = df_characters_movies[df_characters_movies["freebase_ID_actor"].isin(df_actor_id["actor_id"])]
    emotions = ["anger", "disgust", "fear", "joy", "sadness", "surprise"]
    normalized_columns_name_emotion = [f"normalized_plot_{emotion}_without_neutral" for emotion in emotions]

    df_movies_emotions = df_characters_movies[["freebase_ID_actor"] + normalized_columns_name_emotion].set_index("freebase_ID_actor")
    dominant_emotion_per_movie_actor = df_movies_emotions.idxmax(axis=1).str.replace("normalized_plot_", "").str.replace("_without_neutral", "")
    count_actor_emotion = dominant_emotion_per_movie_actor.reset_index().groupby("freebase_ID_actor").value_counts();

    # I tried to set values='count' but it seems to rename all the columns to 'count'
    # Without the values='count' it seems that the columns have the right name but are on two levels
    df_count_actor_emotion = count_actor_emotion.reset_index().pivot_table(index="freebase_ID_actor", columns=0)

    # Not taking the emotions list to make sure that it is the right order
    column_emotion = [count_emotion[1] for count_emotion in df_count_actor_emotion.columns]
    df_count_actor_emotion = df_count_actor_emotion.droplevel(0, axis=1)
    df_count_actor_emotion.columns = column_emotion
    df_count_actor_emotion[df_count_actor_emotion.isna()] = 0
    df_actor_id = df_actor_id.merge(df_count_actor_emotion, left_on="actor_id", right_index=True)

    return df_actor_id