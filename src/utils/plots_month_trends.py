import pandas as pd
import numpy as np
from scipy.stats import pearsonr
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from src.utils.save_plot import save_plot



def plot_variation(type, data, df_movies=None):
    
    """
    This function calculates the mean emotion scores and their percentage variations across predefined 
    periods of the year. The results are visualized using animated bar and scatter plots. The animation 
    displays how emotion scores change over time, providing insights into periodic trends

    Parameters:
    - data_type (str): Specifies the type of data : 'plots' or 'reviews', used to determine the relevant emotion columns
    - data (pd.DataFrame):Dataframe containing the following:
        'release_month': Numeric month of release
        Emotion score columns in the format: 'normalized_<data_type>_<emotion>_without_neutral'
    - df_movies: Dataframe containing movies metadata

    Returns:
    - pd.DataFrame: Dataframe containing percentage variations of emotion scores between consecutive periods

    Outputs:
    - Animated plotly figure showing:
        Mean emotion scores for each period : bar plot
        Percentage variations in scores from the previous period : scatter plot
    """

    if type == "review":
        data=data.select_dtypes(exclude=['object'])
        data = data.groupby(by='wikipedia_ID').mean()
        data = pd.merge(data, df_movies, on='wikipedia_ID', how='inner')

    # type : plots or reviews data
    # Define the periods of year, emotion columns and emojis, emotion colors and display names
    
    periods = {
        "New Year": {"months": [1], "emoji": "🎆🎇", "month_name": "January"},
        "Valentine's Day": {"months": [2], "emoji": "❤️💖", "month_name": "February"},
        "Spring Release": {"months": [3], "emoji": "🌸🌼", "month_name": "March"},
        "Easter Break": {"months": [4], "emoji": "🐣🥚", "month_name": "April"},
        "Pre-Summer Warm-up": {"months": [5], "emoji": "☀️🏖️", "month_name": "May"},
        "Summer movies": {"months": [6, 7], "emoji": "🎥🍿", "month_name": "June/July"},
        "Fall Releases/Back-to-School": {"months": [8,9], "emoji": "📚🍂", "month_name": "August/September"},
        "Halloween": {"months": [10], "emoji": "🎃👻", "month_name": "October"},
        "Thanksgiving": {"months": [11], "emoji": "🦃🍂", "month_name": "November"},
        "Christmas": {"months": [12], "emoji": "🎄🎅", "month_name": "December"}
    }
    
    emotion_labels = ["anger", "disgust", "fear", "joy", "sadness", "surprise"]
    emotion_columns = [f'normalized_{type}_{emotion}_without_neutral' for emotion in emotion_labels]
    emotion_display_names = {f'normalized_{type}_{emotion}_without_neutral': emotion.capitalize() for emotion in emotion_labels}

    emotion_colors = {
        f'normalized_{type}_joy_without_neutral': "#f3c35f",
        f'normalized_{type}_anger_without_neutral': "#e13c29",
        f'normalized_{type}_disgust_without_neutral': "#8bc664",
        f'normalized_{type}_sadness_without_neutral': "#85b8ed",
        f'normalized_{type}_fear_without_neutral': "#b99be5",
        f'normalized_{type}_surprise_without_neutral': "#de9f81"
    }
    
    emotion_emojis = {
        "Anger": "😠",
        "Disgust": "🤢",
        "Fear": "😨",
        "Joy": "😊",
        "Sadness": "😢",
        "Surprise": "😲"
    }

    # Creation of a new column containing the period of release
    data["period"] = data["release_month"].map(
        lambda month: next((key for key, data in periods.items() if month in data["months"]), "Other")
    )

    # Mean of emotions for each period of the year
    periodic_emotions = data.groupby("period")[emotion_columns].mean()

    # Variation of the emotion means in % from the previous period
    # Gives 0% of change for New year as it's the reference period
    periodic_emotions_diff = periodic_emotions.pct_change().fillna(0) * 100
    
    # Calculating the variation of emotion means for the new year from christmas
    periodic_emotions_diff.iloc[0] = (
        (periodic_emotions.iloc[0] - periodic_emotions.iloc[-1]) / periodic_emotions.iloc[-1]
    ) * 100

    # Creation of the barplot graphs an the variations graph
    fig = make_subplots(
        rows=1, cols=2,
        subplot_titles=(None, None),
        specs=[[{"type": "scatter"}, {"type": "bar"}]],
    )
    
    # Creation of frames for each period
    frames = []
    previous_period = 'Christmas'
    for period, details in periods.items():
        if period in periodic_emotions.index:
            # Obtain the informations of the current period
            emotion_means = periodic_emotions.loc[period]
            emotion_variations = periodic_emotions_diff.loc[period]

            # Identify the dominant emotion
            dominant_emotion = emotion_variations.idxmax()
            dominant_color = emotion_colors[dominant_emotion]
            dominant_emotion_name = emotion_display_names[dominant_emotion]

            # Creation of the scatter plots giving the variations in %
            scatter_data = []
            for emotion, value in emotion_variations.items():
                scatter_data.append(go.Scatter(
                    x=[emotion_display_names[emotion]],
                    y=[value],
                    text=[f"<b><span style='color: {'green' if value > 0 else 'red'};'>{value:+.2f}%</span> {emotion_emojis[emotion_display_names[emotion]]}</b>"],
                    mode="markers+text",
                    textposition="bottom center" if value < 0 else "top center",
                    marker=dict(
                        size=abs(value) * 5 + 10, 
                        color=emotion_colors[emotion],
                        symbol="triangle-up" if value > 0 else "triangle-down",
                    ),
                    hovertemplate = "Variation of %{x}: %{y} % <extra></extra>"
                ))
                
            # Creation of barplots with the mean of emotions for the given period 
            barplot_data = go.Bar(
                x=[emotion_display_names[col] for col in emotion_columns],
                y=emotion_means,
                marker=dict(color=[emotion_colors[col] for col in emotion_columns]),
                name=f"Mean score ({details['month_name']})",
                hovertemplate = "Mean score of %{x}: %{y} <extra></extra>"
            )
            frames.append(go.Frame(
                data= scatter_data+[barplot_data],
                name=details["month_name"],  
                layout=go.Layout(
                    annotations=[
                        dict(
                            x=0.47,
                            y=1.31,
                            showarrow=False,
                            xref="paper",
                            yref="paper",
                            text=f"<b>{details['month_name']} : {period} {details['emoji']}<br>Significant variation from last month : {dominant_emotion_name} {emotion_emojis[dominant_emotion_name]}</b>",
                            font=dict(size=18, color=dominant_color, family="Arial, sans-serif")
                        ),
                        dict(
                            text=f"Mean score for each emotion during the current period of the year in {type + 's'}",
                            x=0.98,  
                            y=1.1,  
                            xref="paper",
                            yref="paper",
                            showarrow=False,
                            font=dict(
                                family="Trebuchet MS, sans-serif",  
                                size=14,
                                color="black",
                                weight="bold" 
                            )
                        ),
                        dict(
                            text=f"Period-to-period variation of emotions in {type + 's'}",
                            x=0,  
                            y=1.16,  
                            xref="paper",
                            yref="paper",
                            showarrow=False,
                            font=dict(
                                family="Trebuchet MS, sans-serif",  
                                size=14,
                                color="black",
                                weight="bold"  
                            )
                        ),
                        dict(
                            text=f"Variation trends from {previous_period} to {period}",
                            x=0,  
                            y=1.1,  
                            xref="paper",
                            yref="paper",
                            align="center",
                            showarrow=False,
                            font=dict(
                                family="Trebuchet MS, sans-serif",  
                                size=14,
                                color="black",
                                weight="bold"  
                            ),
                        )
                    ]
                )
            ))
            previous_period = period
            
    # Add the frames to the figure to plot
    initial_frame = frames[0]
    fig.update(frames=frames)

    # Slider 
    fig.update_layout(
        yaxis_title="Mean of emotions",
        yaxis2_title="Variation (%)",
        template="plotly_white",
        showlegend=False,
        sliders=[
            {
                "active": 0,
                "currentvalue": {
                    "prefix": "Month: ",
                    "font": {"size": 15, "color": "black"}
                },
                "steps": [
                    {
                        "args": [[details["month_name"]], {"frame": {"duration": 0, "redraw": True}, "mode": "immediate"}],
                        "label": details["month_name"],  
                        "method": "animate"
                    }
                    for details in periods.values() 
                ]
            }
        ]
    )

    # Axis and legend styling

    fig.update_yaxes(
        range=[-30, 30], 
        title="Variation (%)",
        row=1,
        col=1,
        showticklabels=False
    )
    
    space=[0, 0]
    if type=='review':
        space=[0, 0.41]
    else : 
        space=[0, 0.26] 
            
    fig.update_yaxes(
        range=space,
        row=1,
        col=2
    )
    fig.update_xaxes(
        range=[-0.5, len(emotion_columns) - 0.5], 
        categoryorder='total ascending',  
        row=1,
        col=1,
        title=None
    )

    # Create the initial frame 
    # Add initial traces to the figure
    for trace in initial_frame.data:
        if isinstance(trace, go.Bar):  
            fig.add_trace(trace, row=1, col=2)
        elif isinstance(trace, go.Scatter): 
            fig.add_trace(trace, row=1, col=1)
            
    # Add annotations from the initial frame        
    fig.update_layout(
        annotations=initial_frame.layout.annotations  
    )
           
    frames.insert(0, initial_frame)

    fig.update_layout(
        autosize=True,
        margin=dict(l=50, r=50, t=150, b=50),  
        width= 1200,
        height = 600,
    )
    
    fig.show()
    save_plot(fig, f"variation_{type}")
    return periodic_emotions_diff


def corr_p_value_plot_periods(df1, df2):
    
    """
    This function computes the pearson correlation coefficients and corresponding p-values between emotion scores 
    from two different datasets : in our case plots and reviews. The results are presented in a circular barplot where 
    each bar represents the correlation for a specific emotion. The bar color indicates the significance of the 
    correlation : gray for non-significant results

    Parameters:
    - df1 (pd.DataFrame): The first dataframe containing emotion variations trend over time for the first dataset
    - df2 (pd.DataFrame): The second dataframe containing emotion variations trend over time for the second dataset

    Returns:
    - None: The function directly generates and displays a plotly figure

    Outputs:
    - A circular barplot showing:
        Correlation coefficients for each emotion 
        Color-coded bars indicating significance : gray for p-value ≥ 0.05
    """

    
    # Define the emotion columns and colors and display names
    
    emotion_labels = ["anger", "disgust", "fear", "joy", "sadness", "surprise"]
    emotion_columns_plots = [f'normalized_plot_{emotion}_without_neutral' for emotion in emotion_labels]
    emotion_columns_reviews = [f'normalized_review_{emotion}_without_neutral' for emotion in emotion_labels]
        
    emotion_display_names = {f'normalized_plot_{emotion}_without_neutral': emotion.capitalize() for emotion in emotion_labels}
        
    emotion_colors = {
        'Joy': "#f3c35f",
        'Anger': "#e13c29",
        'Disgust': "#8bc664",
        'Sadness': "#85b8ed",
        'Fear': "#b99be5",
        'Surprise': "#de9f81"
    }
        
    # Gray for non-significant results
    insignificant_color = "#b0b0b0"  

    # Calculate correlations and p-values
    emotions = []
    correlations = []
    p_values = []
    for col_plots, col_reviews in zip(emotion_columns_plots, emotion_columns_reviews):
        values_plots = df1[col_plots].dropna()
        values_reviews = df2[col_reviews].dropna()
        if len(values_plots) >= 2 and len(values_reviews) >= 2:
            corr, p_val = pearsonr(values_plots, values_reviews)
            emotions.append(emotion_display_names[col_plots])
            correlations.append(corr)
            p_values.append(p_val)

    # Define angles and bar widths
    num_emotions = len(emotions)
    angles = np.linspace(0, 2 * np.pi, num_emotions, endpoint=False)  
    bar_width = np.pi / num_emotions  

    # Compute midpoints for aligning labels with bars
    midpoints = angles + (bar_width / 2) 

    # Plotting correlations and p_values in the circular barplot
    fig = go.Figure()

    for i, (angle, corr, p_val, emotion) in enumerate(zip(angles, correlations, p_values, emotions)):
        # Determine bar color based on significance
        color = emotion_colors[emotion] if p_val < 0.05 else insignificant_color
        # Define verticale bars size
        theta = [
            np.degrees(angle),  
            np.degrees(angle + bar_width),  
            np.degrees(angle + bar_width), 
            np.degrees(angle) 
        ]
        r = [0.05, 0.05, corr, corr] 

        fig.add_trace(go.Scatterpolar(
            r=r,
            theta=theta,
            fill='toself',
            name=f'{emotion} ({p_val:.3f})',
            text=f'{emotion}: correlation={corr:.2f}, p_value={p_val:.3f}', 
            hoverinfo="text",
            mode='none', 
            fillcolor=color,  
            opacity=0.8
        ))

    # Axis and legend styling
    fig.update_layout(
        polar=dict(
            angularaxis=dict(
                tickvals=np.degrees(midpoints), 
                ticktext=emotions,  
                rotation=90, 
                direction="clockwise",
                showline=True,
            ),
            radialaxis=dict(
                visible=True,
                range=[0, max(correlations) + 0.1],  
            )
        ),
        title="Correlation and significance analysis of emotional evolution throughout the year between plots and reviews",
        showlegend=True,
        legend=dict(
            title="H0: No correlation between reviews and plots period-to-period emotion variations<br>H0 rejected / correlation coefficient statistically significant : p < 0.05<br>H0 not rejected : p ≥ 0.05<br><br>P-Value significances : <br>",
            x=1.1,
            y=1,
            borderwidth=1,
            bgcolor='rgba(255,255,255,0.7)',
        ),
        template="plotly_white",
    )

    fig.show()
    save_plot(fig, "variation_corr_pvalues")