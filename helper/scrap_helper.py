import requests
from bs4 import BeautifulSoup
import re
from datetime import datetime


def wikidata_from_wikipedia_id(wikipedia_id, language="en"):
    '''
    Returns wikidata link and wikidata id of the given wikipedia id.
    '''
    
    url = f"https://{language}.wikipedia.org/w/api.php"
    params = {
        "action": "query",
        "prop": "pageprops",
        "pageids": wikipedia_id,
        "format": "json"
    }

    response = requests.get(url, params=params)
    if response.status_code == 200:
        data = response.json()
        try:
            wikidata_id = data["query"]["pages"][str(wikipedia_id)]["pageprops"]["wikibase_item"]
            wikidata_link = f"https://www.wikidata.org/wiki/{wikidata_id}"
            return wikidata_link, wikidata_id
        except KeyError:
            # Wikidata ID not found
            return None, None
    else:
        # Error retrieving data
        return None, None
    
def get_release_date(wikidata_id):
    '''
    Returns the release date of the movie with the given wikidata ID.
    '''
    url = f"https://www.wikidata.org/w/api.php"
    params = {
        "action": "wbgetentities",
        "ids": wikidata_id,
        "format": "json"
    }
    
    response = requests.get(url, params=params)
    if response.status_code == 200:
        data = response.json()
        try:
            # Access 'claims' section to find property P577 corresponding to publication date
            publication_claims = data["entities"][wikidata_id]["claims"].get("P577")
            if publication_claims:
                # There can be multiple publication dates but we get the first one
                publication_date = publication_claims[0]["mainsnak"]["datavalue"]["value"]["time"]
                return publication_date
            else:
                # P577 not found
                return None
        except KeyError:
            # Case of unexpected data structure
            return None
    else:
        return None
        # Error retrieving data
        
def format_date_numeric(date_str):
    if date_str == None:
        return '', ''
    year_month = date_str[1:8]  # Extract "+YYYY-MM"
    return year_month[:4], year_month[5:]
    