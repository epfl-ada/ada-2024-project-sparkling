genres_dict = {
    "Action": ["Ninja movie", "Escape Film", "Spy", "Cavalry Film", "Martial Arts Film", "Prison escape", "Gangster Film", "Sword and Sandal", "Swashbuckler films", "Samurai cinema", "Combat Films", "Action/Adventure", "Action Comedy", "Action Thrillers", "Glamorized Spy Film", "Superhero movie", "Sword and sorcery", "Space western", "Adventure", "Outlaw biker film", "Bruceploitation", "Parkour in popular culture", "Chase Movie", "Superhero", "Biker Film", "Auto racing", "Boxing", "Action"],
    "Adventure": ["Coming-of-age film", "Horse racing", "Jungle Film", "Road movie", "Family-Oriented Adventure", "Fantasy Adventure", "Costume Adventure", "Science fiction Western", "Animal Picture", "Extreme Sports", "Fantasy", "Adventure Comedy", "Beach Film", "Travel", "Wuxia", "Foreign legion"],
    "Animated": ["Animated Musical", "Animated cartoon", "Silhouette animation", "Computer Animation", "Stop motion", "Clay animation", "Anime", "Supermarionation", "Animation"],
    "Comedy": ["Comedy-drama", "Slapstick", "Buddy Picture", "Gross out", "Comedy of Errors", "Comedy of manners", "Domestic Comedy", "Comedy film", "Workplace Comedy", "Comedy horror", "Parody", "Courtroom Comedy", "Musical comedy", "Romantic comedy", "Tragicomedy", "Comedy Western", "Heavenly Comedy", "Sex comedy", "Stand-up comedy", "Gross-out film", "Screwball comedy", "Black comedy", "Fantasy Comedy", "Comedy Thriller", "Buddy cop", "Satire", "Comedy Thriller", "Stoner film", "Humour", "Absurdism", "Bloopers & Candid Camera", "Media Satire", "Ealing Comedies", "Beach Party film", "British New Wave", "Comdedy", "Political satire", "Mumblecore", "Slapstick", "Camp", "Mockumentary", "Buddy film", "Female buddy film", "Hip hop movies", "Comedy"],
    "Crime": ["Master Criminal Films", "Crime Fiction", "Crime Drama", "Detective fiction", "Crime", "Neo-noir", "Gangster Film", "Heist", "Law & Crime", "Crime Thriller", "Suspense", "Detective", "Whodunit", "Crime Comedy", "Caper story", "Conspiracy fiction", "Blaxploitation", "Outlaw"],
    "Disaster": ["Natural disaster", "Doomsday film", "Road-Horror", "Disaster", "Nuclear warfare", "Plague"],
    "Documentary": ["Documentary", "Anthropology", "Health & Fitness", "Historical Documentaries", "Rockumentary", "Environmental Science", "World History", "Social issues", "War effort", "Docudrama", "Political Documetary", "Hagiography", "Educational", "Essay Film", "Nature", "Animals", "Archives and records", "Archaeology"],
    "Drama": ["Family Drama", "Biographical film", "Melodrama", "Film à clef", "Marriage Drama", "Psychological thriller", "Kitchen sink realism", "Historical drama", "Political drama", "Romantic drama", "Addiction Drama", "Inspirational Drama", "Social problem film", "Legal drama", "Courtroom Drama", "Psychological thriller", "Slice of life story", "Childhood Drama", "Interpersonal Relationships", "Coming of age", "Tragedy", "Neorealism", "Feminist Film", "Star vehicle", "Americana", "Mumblecore", "Family Film", "Medical fiction", "Journalism", "Ensemble Film", "Family & Personal Relationships", "Race movie", "LGBT", "Biopic [feature]", "Drama", "Christmas movie", "Indie", "Biography", "Private military company", "Women in prison films", "Prison"],
    "Experimental": ["Surrealism", "Avant-garde", "Mondo film", "Absurdism", "Expressionism", "Experimental film", "Existentialism", "Dogme 95", "New Queer Cinema", "Czechoslovak New Wave"],
    "Fantasy": ["Children's Fantasy", "Romantic fantasy", "Fantasy Drama", "Mythological Fantasy", "Sword and sorcery films", "Supernatural", "Alien Film", "Children's Issues", "Heaven-Can-Wait Fantasies", "Space opera", "Alien invasion", "Fantasy Adventure", "Revisionist Fairy Tale", "Fairy tale", "Gothic Film", "Period Horror", "Monster"],
    "Historical": ["Historical fiction", "Historical Epic", "Biographical film", "Costume drama", "Historical Documentaries", "World History", "Period piece", "British Empire Film", "The Netherlands in World War II", "Cold War", "Gulf War", "Early Black Cinema", "History", "Americana", "Film & Television History", "Movies About Gladiators"],
    "Horror": ["Horror", "Sci-Fi Horror", "Psychological horror", "Natural horror films", "Creature Film", "Splatter film", "Demonic child", "Zombie Film", "Monster movie", "Vampire movies", "Werewolf fiction", "Costume Horror", "Road-Horror", "Horror Comedy", "Slasher", "Haunted House Film", "Psycho-biddy", "Giallo", "B-movie", "Cult", "Illnesses & Disabilities", "Monster"],
    "Musical": ["Operetta", "Animated Musical", "Musical comedy", "Backstage Musical", "Courtroom Comedy", "Musical Drama", "Dance", "Jukebox musical", "Film-Opera", "Concert film", "Singing cowboy", "Music", "Instrumental Music", "Musical"],
    "Noir": ["Neo-noir", "Film noir", "Future noir", "Mystery", "Pre-Code"],
    "Pornographic": ["Softcore Porn", "Sexploitation", "Erotica", "Gay pornography", "Hardcore pornography", "Erotic Drama", "Erotic thriller", "Pinku eiga", "Pornographic movie", "Pornography", "Statutory rape", "Gay"],
    "Propaganda": ["Propaganda film", "Patriotic film", "Political cinema", "Social problem film"],
    "Romance": ["Romance Film", "Romantic comedy", "Romantic drama", "Romantic thriller", "Romantic fantasy", "Chick flick", "Gay Themed", "Gay Interest", "LGBT", "Interpersonal Relationships", "Homoeroticism", "Buddy film"],
    "Science Fiction": ["Cyberpunk", "Sci-Fi Horror", "Sci-Fi Thriller", "Time travel", "Alien Film", "Space opera", "Apocalyptic and post-apocalyptic fiction", "Dystopia", "Alien invasion", "Science Fiction", "Steampunk", "Space western", "Sci-Fi Adventure", "Future noir", "Computers", "Tokusatsu", "Z movie", "Kafkaesque", "Sci Fi Pictures original films"],
    "Teen": ["Teen", "Coming-of-age film", "School story", "Juvenile Delinquency Film", "Children's Entertainment", "Children's", "Children's/Family"],
    "Thriller": ["Psychological thriller", "Crime Thriller", "Erotic thriller", "Political thriller", "Action Thrillers", "Suspense", "Thriller", "Erotic thriller", "Conspiracy fiction", "Revenge"],
    "War": ["War film", "Anti-war film", "Combat Films", "Gulf War", "The Netherlands in World War II", "Anti-war"],
    "Western": ["Revisionist Western", "Acid western", "Singing cowboy", "Indian Western", "Hybrid Western", "Epic Western", "B-Western", "Spaghetti Western", "Western", "Outlaw"]
}

import pandas as pd

def get_genres_merged(df):
    '''
    For each genre, create a new column in 'df_merged' with the genre name and set it to 'True'
    if any of the subgenres in 'subgenres' are True in 'df', for each row.
    Arguments:
        df: DataFrame containing subgenres
    Returns:
        df_merged: DataFrame containing genres
    '''
    
    df_merged = pd.DataFrame({'wikipedia_ID': df.wikipedia_ID}) # Create a new DataFrame with the wiki ids.
    for genre, subgenres in genres_dict.items():
        df_merged[genre] = df[subgenres].any(axis=1) # Set genre entry to True each time one of its subgenres is True for a movie.
    return df_merged
