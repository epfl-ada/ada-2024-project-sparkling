import numpy as np
import pandas as pd

def date_choice(dates):
    '''
    Arguments:
        dates: list [year in original dataset, month in original dataset, year scraped, month scraped]
    Returns:
        final_date: tuple with final year and month we chose. One of them could be nan, it
                    will be processed in get_final_dates.
    '''
    year = float(dates[0])
    month = float(dates[1])
    year_scr = float(dates[2])
    month_scr = float(dates[3])
    
    final_date = [0, 0]
    
    if np.isnan(year):
        if np.isnan(month):
            final_date = [year_scr, month_scr]
        else:
            if np.isnan(month_scr):
                final_date = [year_scr, month]
            else:
                if month_scr==1.0:
                    final_date = [year_scr, month]
                else:
                    final_date = [year_scr, month_scr]
    else:
        if year==year_scr:
            if np.isnan(month):
                final_date = [year, month_scr]
            else:
                if np.isnan(month_scr):
                    final_date = [year, month]
                else:
                    if month_scr==1.0:
                        final_date = [year, month]
                    else:
                        final_date = [year, month_scr]
        else:
            if np.isnan(month):
                final_date = [year_scr, month_scr]
            else:
                final_date = [year, month]
                
    return final_date
        
def get_final_dates(df): 
    '''
    Arguments:
        df: dataframe of original dates and scraped dates.
    Returns:
        final_df: dataframe with 3 columns ['wikipedia_ID', 'release_year', 'release_month']
                  with only the movies for which we have found a way to get both their year 
                  and month of release (no NaN).
    '''
    final_df = pd.DataFrame(data={'wikipedia_ID': list(df.wikipedia_ID)})
    
    final_df['release_year'] = df[['release_year_x', 'release_month_x', 'release_year_y', 'release_month_y']].apply(lambda x: date_choice(list(x))[0], axis=1)
    final_df['release_month'] = df[['release_year_x', 'release_month_x', 'release_year_y', 'release_month_y']].apply(lambda x: date_choice(list(x))[1], axis=1)
    
    final_df = final_df.drop((final_df[np.isnan(final_df['release_year']) | np.isnan(final_df['release_month'])]).index)
    
    return final_df