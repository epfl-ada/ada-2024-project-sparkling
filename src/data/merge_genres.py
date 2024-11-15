genres_dict = {
    "Action": ["Ninja movie", "Escape Film", "Spy", "Cavalry Film", "Martial Arts Film", "Gangster Film", "Sword and Sandal", "Swashbuckler films", "Samurai cinema", "Combat Films", "Action/Adventure", "Action Comedy", "Action Thrillers", "Glamorized Spy Film", "Superhero movie", "Sword and sorcery", "Space western", "Adventure", "Outlaw biker film", "Parkour in popular culture", "Chase Movie", "Superhero", "Biker Film", "Auto racing", "Boxing", "Action"],
    "Adventure": ["Coming-of-age film", "Horse racing", "Jungle Film", "Road movie", "Family-Oriented Adventure", "Fantasy Adventure", "Costume Adventure", "Science fiction Western", "Animal Picture", "Extreme Sports", "Fantasy", "Adventure Comedy", "Beach Film", "Travel", "Wuxia", "Foreign legion"],
    "Animated": ["Animated Musical", "Animated cartoon", "Computer Animation", "Stop motion", "Clay animation", "Anime", "Supermarionation", "Animation"],
    "Comedy": ["Comedy-drama", "Slapstick", "Gross out", "Comedy of Errors", "Comedy of manners", "Domestic Comedy", "Comedy film", "Workplace Comedy", "Comedy horror", "Parody", "Courtroom Comedy", "Musical comedy", "Romantic comedy", "Tragicomedy", "Comedy Western", "Heavenly Comedy", "Sex comedy", "Stand-up comedy", "Gross-out film", "Screwball comedy", "Black comedy", "Fantasy Comedy", "Comedy Thriller", "Buddy cop", "Satire", "Comedy Thriller", "Stoner film", "Humour", "Absurdism", "Bloopers & Candid Camera", "Media Satire", "Ealing Comedies", "British New Wave", "Comdedy", "Political satire", "Mumblecore", "Slapstick", "Camp", "Mockumentary", "Buddy film", "Female buddy film", "Hip hop movies", "Comedy"],
    "Crime": ["Master Criminal Films", "Crime Fiction", "Crime Drama", "Detective fiction", "Crime", "Neo-noir", "Gangster Film", "Heist", "Law & Crime", "Crime Thriller", "Suspense", "Detective", "Whodunit", "Crime Comedy", "Caper story", "Conspiracy fiction", "Blaxploitation", "Outlaw"],
    "Disaster": ["Natural disaster", "Doomsday film", "Road-Horror", "Disaster", "Nuclear warfare", "Plague"],
    "Documentary": ["Documentary", "Rockumentary", "World History", "Social issues", "Docudrama", "Hagiography", "Educational", "Essay Film", "Nature", "Animals", "Archives and records", "Archaeology"],
    "Drama": ["Family Drama", "Biographical film", "Melodrama", "Film à clef", "Marriage Drama", "Psychological thriller", "Kitchen sink realism", "Historical drama", "Political drama", "Romantic drama", "Addiction Drama", "Inspirational Drama", "Social problem film", "Legal drama", "Courtroom Drama", "Psychological thriller", "Slice of life story", "Childhood Drama", "Interpersonal Relationships", "Coming of age", "Tragedy", "Feminist Film", "Star vehicle", "Americana", "Mumblecore", "Family Film", "Medical fiction", "Journalism", "Ensemble Film", "Family & Personal Relationships", "Race movie", "LGBT", "Biopic [feature]", "Drama", "Christmas movie", "Indie", "Biography", "Private military company", "Women in prison films", "Prison"],
    "Experimental": ["Surrealism", "Avant-garde", "Mondo film", "Absurdism", "Expressionism", "Experimental film", "Existentialism", "Dogme 95", "Czechoslovak New Wave"],
    "Fantasy": ["Children's Fantasy", "Romantic fantasy", "Fantasy Drama", "Mythological Fantasy", "Sword and sorcery films", "Supernatural", "Alien Film", "Heaven-Can-Wait Fantasies", "Space opera", "Alien invasion", "Fantasy Adventure", "Revisionist Fairy Tale", "Fairy tale", "Gothic Film", "Period Horror", "Monster"],
    "Historical": ["Historical fiction", "Historical Epic", "Biographical film", "Costume drama", "World History", "Period piece", "British Empire Film", "The Netherlands in World War II", "Cold War", "Gulf War", "Early Black Cinema", "History", "Americana", "Movies About Gladiators"],
    "Horror": ["Horror", "Sci-Fi Horror", "Psychological horror", "Natural horror films", "Creature Film", "Splatter film", "Demonic child", "Zombie Film", "Monster movie", "Vampire movies", "Werewolf fiction", "Costume Horror", "Road-Horror", "Horror Comedy", "Slasher", "Haunted House Film", "Psycho-biddy", "Giallo", "B-movie", "Cult", "Illnesses & Disabilities", "Monster"],
    "Musical": ["Operetta", "Animated Musical", "Musical comedy", "Backstage Musical", "Courtroom Comedy", "Musical Drama", "Dance", "Jukebox musical", "Concert film", "Singing cowboy", "Music", "Instrumental Music", "Musical"],
    "Noir": ["Neo-noir", "Film noir", "Future noir", "Mystery", "Pre-Code"],
    "Pornographic": ["Softcore Porn", "Sexploitation", "Erotica", "Gay pornography", "Hardcore pornography", "Erotic Drama", "Erotic thriller", "Pinku eiga", "Pornographic movie", "Pornography", "Gay"],
    "Propaganda": ["Propaganda film", "Political cinema", "Social problem film"],
    "Romance": ["Romance Film", "Romantic comedy", "Romantic drama", "Romantic fantasy", "Gay Themed", "Gay Interest", "LGBT", "Interpersonal Relationships", "Homoeroticism", "Buddy film"],
    "Science Fiction": ["Cyberpunk", "Sci-Fi Horror", "Sci-Fi Thriller", "Time travel", "Alien Film", "Space opera", "Apocalyptic and post-apocalyptic fiction", "Dystopia", "Alien invasion", "Science Fiction", "Steampunk", "Space western", "Sci-Fi Adventure", "Future noir", "Computers", "Z movie", "Kafkaesque", "Sci Fi Pictures original films"],
    "Teen": ["Teen", "Coming-of-age film", "School story", "Juvenile Delinquency Film", "Children's", "Children's/Family"],
    "Thriller": ["Psychological thriller", "Crime Thriller", "Erotic thriller", "Political thriller", "Action Thrillers", "Suspense", "Thriller", "Erotic thriller", "Conspiracy fiction"],
    "War": ["War film", "Anti-war film", "Combat Films", "Gulf War", "The Netherlands in World War II", "Anti-war"],
    "Western": ["Revisionist Western", "Acid western", "Singing cowboy", "Indian Western", "Hybrid Western", "Epic Western", "B-Western", "Spaghetti Western", "Western", "Outlaw"]
}

import pandas as pd

def get_genres_merged(df):
    '''
    Create a simplified DataFrame by grouping subgenre columns into broader genre categories.
    Each broader genre will have its own column, set to `True` if any of its corresponding subgenres 
    in the original DataFrame are `True`.
    
    Arguments:
        df: DataFrame containing subgenres
    Returns:
        df_merged: DataFrame containing genres
    '''
    
    df_merged = pd.DataFrame({'wikipedia_ID': df.wikipedia_ID}) # Create a new DataFrame with the wiki ids.
    for genre, subgenres in genres_dict.items():
        df_merged[genre] = df[subgenres].any(axis=1) # Set genre entry to True each time one of its subgenres is True for a movie.
    return df_merged

