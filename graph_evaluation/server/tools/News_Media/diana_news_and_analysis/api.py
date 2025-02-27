import requests
import json
from datetime import date, datetime, timedelta
import os

from typing import Optional, Dict, Union, List


def counts(date: int, time_period: str, search_term: str, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Lists the number of articles by day, hour, or minute for the search term."
    date: Four digit year, two digit month, and two digit day
        time_period: One of daily, hourly, or minutes.
        search_term: Search term or keyword
        
    """
    url = f"https://diana-news-and-analysis.p.rapidapi.com/counts/{time_period}/{date}/{search_term}"
    querystring = {}
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "diana-news-and-analysis.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def date(date: int, search_term: str, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Lists all of the dates that have an article match for a search term.  These results can then be used for the other access points."
    date: Four digit year and two digit month
        date: Four digit year and two digit month
        search_term: Search term or keyword
        search_term: Search term or keyword
        
    """
    url = f"https://diana-news-and-analysis.p.rapidapi.com/dates/{date}/{search_term}"
    querystring = {}
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "diana-news-and-analysis.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def articles(date: str, search_term: str, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Lists all of the articles and scores for one or more search terms.  This endpoint accepts up to four search terms.  Search terms should be added to the end of the request string with a forward slash (e.g., /articles/20190101/search-term-one/search-term-two/search-term-three/search-term-four)"
    date: Four digit year, two digit month, and two digit day
        search_term: Search term or keyword
        
    """
    url = f"https://diana-news-and-analysis.p.rapidapi.com/articles/{date}/{search_term}"
    querystring = {}
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "diana-news-and-analysis.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

