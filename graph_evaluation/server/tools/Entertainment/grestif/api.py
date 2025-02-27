import requests
import json
from datetime import date, datetime, timedelta
import os

from typing import Optional, Dict, Union, List


def search_gifs(q: str, limit: str='20', toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Search gives you instant access to our library of millions of GIFs and Stickers by entering a word or phrase. Get a json object containing a list of alternative search terms given a search term."
    q: Mention your search query as string
        
    """
    url = f"https://grestif.p.rapidapi.com/search"
    querystring = {'q': q, }
    if limit:
        querystring['limit'] = limit
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "grestif.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def trending_search_words(toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Provides users a list of the most popular trending search terms. Get a json object containing a list of the current trending search terms."
    
    """
    url = f"https://grestif.p.rapidapi.com/trendingSearches"
    querystring = {}
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "grestif.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def trending_gifs(limit: str='20', toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Trending returns a list of the most relevant and engaging content each and every day. Get a json object containing a list of the current global trending GIFs. The trending stream is updated regularly throughout the day."
    limit: Limit describes the number of results you want to get
        
    """
    url = f"https://grestif.p.rapidapi.com/trending"
    querystring = {}
    if limit:
        querystring['limit'] = limit
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "grestif.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

