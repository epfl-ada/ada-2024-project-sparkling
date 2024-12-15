---
layout: default
title: "From Screen to Heart : How do Emotions Shape the Cinematic Landscape?"
---

{% include background-header.html %}


# Introduction {#Introduction}

<p style="text-align: justify;">&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;Cinema is more than entertainment, it’s an emotional symphony that grips the soul, stirs the heart, and leaves a mark long after the screen fades to black or the popcorn runs out. It’s <b>Joy</b> that makes us laugh, <b>Fear</b> that keeps us on edge, and <b>Sadness</b> that moves us to tears. But why do certain films resonate so deeply? How do they manage to make us feel as though they were crafted just for us? This emotional power is no accident, it’s the result of a masterful design. Behind every unforgettable moment lies a deliberate strategy, where the film industry transforms emotions into tools to shape genres, guide storytelling, and create experiences that resonate strongly with audience expectations.</p>

<p style="text-align: justify; border-left: 3px solid #cccccc; padding: 10px 15px; margin: 15px 0; background-color: #f9f9f9; border-radius: 5px;"><b>This leads us to an essential question: how does the big screen industry leverage emotions to shape the cinematic landscape and redefine the audience experience?</b></p>

<p style="text-align: justify;">To crack this mystery, we’re heading straight into the Emotion Lab. And guess what? We’ve got the best experts on the job. Who better to call on than the ultimate dream team : <b>Joy</b>, <b>Sadness</b>, <b>Fear</b>, <b>Anger</b>, <b>Disgust</b>, and, of course, the mADAjestic <b>Surprise</b>. These InFweside Out insiders are ready to lead the charge, running experiments, designing emotional blueprints, and uncovering how cinema pulls the strings of our hearts. So grab your popcorn (and maybe a tissue), and let’s see how emotions turn the magic of the silver screen into something unique.</p>

## Our experts {#Our-experts}

{% include photo-section.html %}

## Let’s dive deeper into our brain {#Lets-dive-deeper-into-our-brain}

<p style="text-align: justify;">What resources do we have at our disposal?</p>

{% include resources.html %}

<p style="text-align: justify;">We could not have conducted our experiments without the help of some remarkable <b>mathematical tools!</b> Techniques such as Pearson Correlation, Spearman Correlation, Principal Component Analysis (PCA), and K-Means Clustering played a crucial role in shaping our analyses. Click on our <b>memory orbs</b>:</p>

{% include stats.html %}

## What is our DistilRoberta model capable of? {#What-is-our-DistilRoberta-model-capable-of}

<p style="text-align: justify;">The text below is the Inside Out movie plot. Each sentence is highlighted in the color of the emotion that has been classified with the highest score by our model. The sentences that are assigned no color have been classified as neutral, which means that it doesn’t have a predominant emotion.</p>

{% include example_plot.html %}

<p style="text-align: justify;">And if you want to know how our model define each emotion based on the movie’s plots, let’s take a look at these spectacular wordclouds that beautifully align with our experts shapes:</p>

{% include emotions_plots_wordcloud.html %}

# How are our movies classified? {#How-are-our-movies-classified}

<p style="text-align: justify;">Intuitively, we would all think that movie emotions are closely tied to the film's genre. So, we’d like to see how emotions are conveyed based on genres. Let’s first take a look at how many genres we have at our disposal!</p>

{% include genres_surprise.html %}

<p style="text-align: justify;"><b>Surprise</b> is right, to simplify this complex view, we grouped genres into broader categories, making trends easier to study and interpret.</p>

{% include genres_joy.html %}

<p style="text-align: justify;">You're probably wondering why our experts rushed to analyze genres when our main focus is emotions. Well, let them clarify it and shed light on the correlation between genres and emotions.</p>

## Correlation between genres and emotions {#Correlation-between-genres-and-emotions}

<p style="text-align: justify;">If you’re curious about what this figure is, we're here to elucidate the mystery. This is a heatmap, each i-th row of emotion and j-th column genre corresponds to the correlation between both. Therefore, it reveals how each emotion naturally aligns with specific film genres based on the narratives present in the plots.</p>

{% include corr_genres_emotions.html %}

<p style="text-align: justify;">Each emotion instinctively knows where it can shine, and where it must fade, orchestrating the cinema to illuminate the greatest scenes.</p>

<p style="text-align: justify;">Additionally, the p-values confirm that the strong, positive, or negative correlations observed in the heatmap are not merely intuitive but statistically significant. They also reveal non existent emotion-genre relationships (p ≥ 0.05, near-zero correlations), showing that some emotions remain foreign to specific genres, although less pronounced than if the correlation were negative.</p>

{% include_relative plots/emotion_genre_heatmap_p_val.html %}

{% include pvalues_genres_emotions.html %}

<p style="text-align: justify;">This suggests that the emotions audiences commonly associate with certain genres are not always directly conveyed through the plots or film narratives but instead emerge from how viewers perceive and interpret the stories thanks to the filmmaker jobs.</p>

## Emotion distribution of movie plots across genres {#Emotion-distribution-of-movie-plots-across-genres}

<p style="text-align: justify;">Alright, now that we are sure the we have a rock-solid foundation for our analysis, it’s time we dived into the fun part : exploring how emotions are sprinkled across different genres. Let’s see where <b>Joy</b>, <b>Fear</b>, and all their buddies decide to hang out!</p>

{% include_relative plots/Emotion_Distribution_of_Movie_Plots_Across_Genres.html %}

{% include_relative plots/Emotion_Distribution_of_Movie_Reviews_Across_Genres.html %}

<p style="text-align: justify;">These plots give us the overall distribution of emotions across genres, let’s take a deeper dive into how specific emotions vary within movie plots and reviews. By breaking down each emotion, we can uncover fascinating trends and nuances - like how the emotions are differently scaled between plots and reviews.</p>

<p style="text-align: justify;">Click on your favorite emotion! You will see the scores of this emotion in movie's plots and reviews by genres:</p>

{% include emotions_plots_genres.html %}

## Clustering of movies based on their mean plot emotions

<p style="text-align: justify;">But now, can we find different types of movies based on their plot and emotional profile ? Let's run a clustering algorithm !<br>We ran Kmeans with two clusters based on the emotions extracted from the plot of each movie based on the silhouette score.<br>Let’s now take a look at the mean emotion of genres in each of the clusters.</p>

{% include_relative plots/percentage_movie_genre_plot.html %}

<p style="text-align: justify;">As we can see, two different families of movie genres are emerging. The first cluster, “Heart island”, is the funny-romance cluster, centered around comedy and romance movies. The second one, "Shadow Island", comprises movies that relate action, some mystery, and crime without forgetting a bit of comedy as the cherry on top.<br>Our experts like to observe them from time to time.</p>

<div style="text-align: center;">
  <img src="images/islands.jpg" alt="Islands" style="max-width: 30%; transition: transform 0.2s" onmouseover="this.style.transform='scale(1.05)';" onmouseout="this.style.transform='scale(1)';">
</div>

<p style="text-align: justify;">But now let's see if we can find the same type of movies in the emotional profiles of movies based on their reviews.<br>So once again, let's cluster our movies, and using the silhouette score, we have two clusters emerging.</p>

{% include_relative plots/percentage_movie_genre_review.html %}

<p style="text-align: justify;">Interestingly, similar clusters seem to appear once again. But let us confirm our intuition using t-test analysis. Through Pearson’s and Spearman’s correlation, we can see our groups are correlated with a correlation coefficient above 0.95 and a pvalues below 1e-12 for both clusters of genres.</p>

<p style="text-align: justify;">It is interesting to see that we seem to have two types of movies, with emotional profiles reflected in both the plots and the reviews left by the viewers.</p>

# Emotion transitions throughout movies {#Emotion-transitions-throughout-movies}

<p style="text-align: justify;">Now that we’ve explored how emotions are distributed across different genres, why not delve further into our analysis. Have you ever felt, while watching a movie, that every genre seems to follow its own pre-defined emotional flow? Oh, those filmmakers, always trying to toy with our emotions, right?</p>

<p style="text-align: justify;">As long as we’re still in the lab, let’s let our expert team take over. Who better than <b>Joy</b>, <b>Sadness</b>, and the rest of the gang to break down these transitions and reveal the emotional blueprints that guide us through every movie? And it’s simple! Our experts divided each film into three parts:</p>

- Beginning : representing the first quarter of the movie  
- Middle : encompassing the next two quarters  
- End : capturing the final quarter

<p style="text-align: justify;">Ready to share their findings, our expert team rolled out the projector and proudly displayed the results. So, let’s uncover the emotional journey scene by scene and see what they’ve found!</p>

## Heatmap {#Heatmap}

{% include heatmap_with_disgust.html %}

<p style="text-align: justify;">Here, each line corresponds to the probability distribution of the transitions from an emotion. So each row and column correspond to the probability of transitioning to the j-th column knowing that we are in the i-th row. As we can see in the legend, the darker the color is, the higher the probability.</p>

<p style="text-align: justify;">Let’s take some genres examples and see what this weird green square tells us:</p>

{% include heatmap_interpretations.html %}

## Sankey Diagram {#Sankey-Diagram}

<p style="text-align: justify;">Let’s now take a closer look at how emotions transition within the films themselves. We have a stunning Sankey chart, full of vibrant colors!</p>

{% include genres_sankey_plot.html %}

<p style="text-align: justify;">This analysis reveals how the film industry not only uses emotions to tell stories but also to shape and define genres themselves. Each genre is uniquely distinguished, not just by its overall emotional tone, but by the transitions that weave one feeling into another. From the interplay of joy and sadness in romances to the dominance of fear and surprise in war films, these patterns highlight how emotions give genres their unique identity. Rather than serving as a mere backdrop to the narrative, emotions are wielded as a defining force, shaping genres into powerful tools that influence how we connect with films.</p>

<p style="text-align: justify;">Moreover, the behavior of emotions within genres closely mirrors real life, which these genres are meant to illustrate. Whether it’s the comfort of joy in comedies, the suspenseful fear of thrillers, or the bittersweet sadness of dramas, these emotional dynamics reflect the complexities of human experience. This purposeful orchestration ensures that every genre offers its own distinct emotional rhythm, not only captivating audiences but also resonating with the realities they live, leaving them immersed in unforgettable cinematic experiences.</p>

{% include sankey_interpretations.html %}

## Emotion evolution and variations throughout a movie {#Emotion-evolution-and-variations-throughout-a-movie}

<p style="text-align: justify;">What if we dive deeper into variations of emotions ? Not only their distributions. It could be interesting to see how each emotion proportion increases or decreases throughout a movie, and which ones vary the most. We used two types of plots : a barplot and a scatter plot. The bar plot gives the emotion distribution of each time slot whereas the scatter plot translates the emotion variations between the actual time slot and the previous one.</p>

{% include genres_bar_scatter_plot.html %}

<p style="text-align: justify;">Let’s look closer at some genres.</p>

{% include variations_throughout_interpretations.html %}

# Emotion trends over year periods {#Emotion-trends-over-year-periods}

<p style="text-align: justify;">We’ve spent some time diving into the details and exploring how emotions flow within films, but what if we zoom out for a broader view? 
Have you  ever noticed how every time of year comes with its own emotional vibe? <b>Fear</b> takes over during Halloween, joy sparkles at Christmas, and heartbreak might just steal the show on Valentine’s Day. But do you think the film industry taps into these seasonal moods as part of a calculated marketing strategy?
Let’s go back into the Emotion Lab and see how the calendar shapes the emotional journeys we experience on the big screen!"</p>

<iframe src="plots/variation_plot.html" style="width: 100%; height: 600px; border: none;"></iframe>

<p style="text-align: justify;">The month-by-month variation in plot emotions tells a fascinating story about how sentiments evolve with the year periods. Each trend seems indeed synchronized with key moments of the year, reflecting a close link between the film industry and the emotions highlighted during the year in our real life.</p>

{% include trends_over_year_interpretations.html %}

<p style="text-align: justify;">These trends clearly show how the film industry adjusts its stories to the different times of year. Each variation is not insignificant: it echoes the moments shared by audiences at different times of the year, confirming that cinema is an emotional mirror, perfectly tuned for each season.</p>

{% include anger_experiment.html %}

<iframe src="plots/variation_review.html" style="width: 100%; height: 600px; border: none;"></iframe>

<p style="text-align: justify;">In the reviews, joy explodes over the New Year period, followed by the rise of <b>Disgust</b> in February, mirroring the emotions of the plots. In March, there’s a noticeable rise in joy, linked to the arrival of spring, but from here the differences with the plots then widen. During June and July, joy remains dominant, unlike the plots where more varied emotions take over. <b>Sadness</b> makes its appearance in August and September, probably influenced by the end of summer and the return to routine, while in October, <b>Surprise</b> gains ground with Halloween, as in the plots. At Thanksgiving, <b>Sadness</b> and <b>Surprise</b> mix, before <b>Joy</b> triumphs in December, driven by the festive spirit.</p>

<p style="text-align: justify;">These divergences show that viewers are more likely to express their overall feelings, influenced by the season, the general atmosphere of the movie and the overall cinematic experience, rather than by the intrinsic and complex emotions of the stories.</p>

{% include_relative plots/variation_corr_pvalues.html %}

<p style="text-align: justify;">This graph emphasizes the differences in period-to-period emotional trends between plots and reviews throughout the year. Among all analyzed emotions, only <b>Disgust</b> shows a significant correlation (p = 0.020), indicating consistency between the two sources for this specific emotion. The lack of significance for other emotions reflects divergences in how viewers perceive emotions compared to those expressed in film narratives over the months.</p>

# Time for holidays! {#Time-for-holidays}

{% include holidays.html %}

## Emotion map of movie plots {#Emotion-map-of-movie-plots}

<p style="text-align: justify;">On the following graph, we can explore the most dominant emotion in movie plots across decades and countries.</p>

{% include_relative plots/emotion_map_final.html %}

<p style="text-align: justify;">Ohhh! It looks like the world has been screaming anger, sadness, and disgust…</p>

<p style="text-align: justify;">The United States of America, for instance, starts by having mostly sad movies released until 1950, then it enters into a big spiral of anger.</p>

{% include joy_is_sad.html %}

<p style="text-align: justify;">But does this mean that the perception of the world is also as gloomy and pessimistic for the reviewers of these movies?<br>Let’s find out!</p>

## Emotion map of reviews {#Emotion-map-of-reviews}

<p style="text-align: justify;">This time, we’ll map the most dominant emotion of the reviews across decades and countries.</p>

<p style="text-align: justify;">Are you ready for this new adventure? Let’s go!</p>

{% include_relative plots/reviews_map_final.html %}

<img src="images/joy_thoughtful.jpeg" alt="Joy houghtful" style="max-width: 30%; transition: transform 0.2s" onmouseover="this.style.transform='scale(1.05)';" onmouseout="this.style.transform='scale(1)';">

<p style="text-align: justify;">Ah! That’s way better. Look at this beautiful gold color!</p>

<p style="text-align: justify;">We can clearly see that it’s not because a country is more represented by angry, sad, scary, or disgusted movies that the watcher considers it this way.</p>

<p style="text-align: justify;">In 1950, movies across all countries bring joy to the reviewers. And if you scroll through time, you’ll notice something remarkable, reviews around the world are generally filled with joy. Isn’t that fantastic? A universal spark of happiness, transcending borders and eras! But sometimes, emotions like disgust and fear can also dominate reviews in some countries. Take Peru in 1970 or Russia in 2010, for instance, where fear and disgust dominate both plots and reviews.</p>

<p style="text-align: justify;">What does this tell us about the average rating of these movies per country? Is it because the most dominant emotion is joy that the average rating for countries will be high all over the world too? Let’s investigate!</p>

{% include_relative plots/ratings_map_final.html %}

<p style="text-align: justify;">Notice that the bubble’s size is correlated to the number of movies released in the country.</p>

- **USA**: With over 8,000 movies, it contains more than 60% of our dataset, averaging 6.3 in rating.
- **UK and India**: Both have a similar average (6.3) despite their smaller movie counts (1,586 and 493, respectively).
- **Canada**: With 483 movies, also follows the same trend.

<p style="text-align: justify;">Interestingly, the trend in average ratings seems to generally follow the wave of joy's dominance in reviews across all these countries, suggesting that experiencing this emotion after watching a film significantly influences how positively audiences evaluate it.</p>

<p style="text-align: justify;">But despite this overall trend, there are some notable exceptions like Zambia and Zimbabwe in which the movies are rated poorly but still show joy as the dominant emotion in reviews. This might be due to the lack of advanced cinematic infrastructure or resources in these regions, making it challenging to fully entertain and satisfy the audience.</p>

## Ratings per reviews emotion {#Ratings-per-emotion}

<p style="text-align: justify;">Let’s analyze this deeper by plotting statistics on the ratings per dominant emotion across all movies.</p>

{% include ratings_per_emotions.html %}

<p style="text-align: justify;">Oh, and remember the emotion transition graph we looked at earlier? It showed how emotions evolve from the beginning to the middle and then to the end. <b>Joy</b>, while nearly absent in the middle, makes a non-negligible comeback in the final act, maybe another marketing strategy to ensure audiences leave the theater on a better feeling and thus let out a better rate.</p>

## Emotions per rating {#Emotion-per-rating}

<p style="text-align: justify;">And now let's analyze the other way around! On the following graphs we can explore the emotion distribution of movie's plots and reviews accross different average movie rating bins.</p>

{% include_relative plots/Emotion_Distribution_of_Movie_Plots_by_Average_Movie_Rating_Bins.html %}

{% include_relative plots/Emotion_Distribution_of_Movie_Reviews_by_Average_Movie_Rating_Bins.html %}

<p style="text-align: justify;">And now click on your favorite emotion! You will see more into details the best or the worst rate a film has if its plot or its reviews represent this emotion.</p>

{% include emotions_plots_ratings.html %}

# Conclusion {#Conclusion}

<p style="text-align: justify;">We have shown how the movie industry masterfully leverages emotions, not only to tell stories but to define and reshape genres themselves. Each emotion, brought to life by our experts <b>Joy</b>, <b>Sadness</b>, <b>Fear</b>, <b>Anger</b>, <b>Disgust</b> and <b>Surprise</b>, plays a distinct role in crafting unforgettable cinematic experiences. Each genre finds its unique identity through a deliberate orchestration of emotions, whether the comforting joy of comedies, the gripping fear of thrillers or the poignant sadness of dramas. 
<b>Joy</b> lights up the screen in romances and musicals, <b>Fear</b> dominates the suspenseful twist of thrillers. <b>Sadness</b> quietly thrives in dramas, <b>Anger</b> fuels the energy and intensity of actions and crime narratives, <b>Disgust</b> unsettles viewers in experimental and horror movies and <b>Surprise</b> keeps audiences on the edge of their seats in science fiction and adventures. Our emotional experts weave intricate transitions, transforming genres into emotional symphonies that captivate audiences and resonate deeply with their personal experiences. It reveals how filmmakers strategically craft emotional journeys to leave lasting impressions on us. This design ensures that our experts not only enrich the storytelling, but also create some unforgettable cinematic experiences and leave audiences immersed in the magic of the silver screen!</p>

# Play time! {#Play-time}

<p style="text-align: justify;">The core of our analysis may be wrapped up, but the fun is far from over! Think of this as the post-credits scene, a chance to explore some lighter, unexpected insights and entertaining discoveries. Sit back and enjoy as we dive into these bonus highlights and surprising experiments!</p>

## Shining the spotlight: Actors {#Shining-the-spotlight}

<p style="text-align: justify;">Actors are the heart and soul of every movie. And yet, we haven’t talked about them. Let’s delve into our dataset and uncover what the profiles of these cinematic storytellers reveal.
We used the Principal Component Analysis algorithm to display our actors according to the mean emotions of the movies they played in, computed on the plots and the reviews, respectively.</p>

{% include 3d_graph.html %}

{% include pizza_graph.html %}

### Actor emotional profile selector {#Actor-emotional-profile-selector}

<p style="text-align: justify;">Let's see in more details the emotional profiles of our main actors. We comprised, in the following plot, the top 100 actors in terms of the number of movies played in our data set. </p>

{% include_relative plots/actor_selector_emotions_profile.html %}

<p style="text-align: justify;">So, now the question that you may have on your lips. Who are those that make us feel the most emotional?</p>

{% include actors_emotions.html %}

## Recommender System {#Recommender-System}

<p style="text-align: justify;">Our journey may have come to an end, but we’re sure you’re not tired of us yet! Here’s a fun little recommender for you to explore and enjoy more movies.</p>

{% include recommender.html %}

# References

<p style="text-align: justify;">- Some images on this website have been generated by <a href="https://openai.com/dall-e" target="_blank">DALL·E</a>.<br>- The images on the actor emotional profile selector are taken from the corresponding <a href="https://www.wikipedia.org/" target="_blank">Wikipedia</a> page.<br>- The movie posters on the recommender system was taken from the <a href="https://www.themoviedb.org/" target="_blank"> TMDB</a> API.<br>- The word clouds graphs have been generated in Python using the <a href="https://pypi.org/project/wordcloud/" target="_blank">wordcloud</a> library.<br>- All the other graphs have been generated in Python using the <a href="https://plotly.com/python/" target="_blank">Plotly</a> library.<br>- Some illustrating images came from the movie <i>Inside Out</i> and have been downloaded from Internet.</p>



<script>
  window.onload = function() {
    showGraphWordCloud('Joy');
    showGraphCorrGenres('Joy')
    showGraphEmotionsGenres('Joy');
    const firstHeatmapButton = document.querySelector('.heatmap-buttons button');
    if (firstHeatmapButton) {
        showHeatmapDescription(firstHeatmapButton.textContent.trim(), firstHeatmapButton);
    }
    const firstSankeyButton = document.querySelector('.sankey-buttons button');
    if (firstSankeyButton) {
        showSankeyDescription(firstSankeyButton.textContent.trim(), firstSankeyButton);
    }
    const firstVariationsThroughoutButton = document.querySelector('.buttons-variations-throughout button');
    if (firstVariationsThroughoutButton) {
        showVariationsThroughoutInterpretation(firstVariationsThroughoutButton.innerText, firstVariationsThroughoutButton);
    }
    loadGenresHeatmapGraph('All Genres');
    loadGenresSankeyGraph('All Genres');
    loadGenresScatterGraph('All Genres');
    showGraphEmotionRatings('Joy');
  };
</script>

