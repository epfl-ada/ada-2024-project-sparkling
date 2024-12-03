---
layout: default
title: "From Screen to Heart : How do Emotions Shape the Cinematic Landscape?"
---

{% include background-header.html %}

# Introduction {#Introduction}

Blablabla ici on va décrire un peu une introduction, pour introduire tu connais
Blablabla ici on va décrire un peu une introduction, pour introduire tu connais
Blablabla ici on va décrire un peu une introduction, pour introduire tu connais
Blablabla ici on va décrire un peu une introduction, pour introduire tu connais

{% include photo-section.html %}

# Emotions {#Emotions}

Ici on parle des émotions, de comment on a produit le dataset, avec qq plots et/ou reviews en exemple avec le texte coloré dans la couleur de sa main emotion.

# Genres {#Genres}

Ici on va parler des genres takapté.

Ce serait bien de mettre qq genres où quand on clique ça affiche un exemple de film de ce genre.

## Emotion Distribution Across Genres {#Emotion-Distribution-Across-Genres}

Dans cette section on va voir les emotions par genre tu coco

### Emotion Distribution of Movie Plots Across Genres {#Emotion-Distribution-of-Movie-Plots-Across-Genres}

On the following graph we can explore the emotion distribution of movie plots accross different genres

{% include_relative plots/Emotion_Distribution_of_Movie_Plots_Across_Genres.html %}

### Emotion Distribution of Movie Reviews Across Genres {#Emotion-Distribution-of-Movie-Reviews-Across-Genres}

On the following graph we can explore the emotion distribution of movie reviews accross different genres

{% include_relative plots/Emotion_Distribution_of_Movie_Reviews_Across_Genres.html %}

### Analyse your emotion {#Analyze-Genres}

Là tu peux cliquer sur l'émotion que tu veux pour avoir les 2 graphiques spécifiques et une interprétation personalisé tu connais

{% include emotions_plots_genres.html %}

# Ratings {#Ratings}

Là c'est l'endroit des ratiiiings (t un 10/10)

## Emotion Distribution by Average Movie Rating {#Emotion-Distribution-by-Average-Movie-Rating}

Dans cette section on va voir les emotions par ratings tu coco

### Emotion Distribution of Movie Plots by Average Movie Rating {#Emotion-Distribution-of-Movie-Plots-by-Average-Movie-Rating}

On the following graph we can explore the emotion distribution of movie plots accross different average movie rating bins

{% include_relative plots/Emotion_Distribution_of_Movie_Plots_by_Average_Movie_Rating_Bins.html %}

### Emotion Distribution of Movie Reviews by Average Movie Rating {#Emotion-Distribution-of-Movie-Reviews-by-Average-Movie-Rating}

On the following graph we can explore the emotion distribution of movie reviews accross different average movie rating bins

{% include_relative plots/Emotion_Distribution_of_Movie_Reviews_by_Average_Movie_Rating_Bins.html %}

### Analyse your emotion {#Analyze-Ratings}

Là tu peux cliquer sur l'émotion que tu veux pour avoir les 2 graphiques spécifiques et une interprétation personalisé tu connais

{% include emotions_plots_ratings.html %}





<script>
// Afficher le premier graphique par défaut au chargement de la page
window.onload = function() {
showGraphGenres('Joy');
showGraphRatings('Joy');
};
</script>

