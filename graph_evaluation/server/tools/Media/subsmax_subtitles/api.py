import requests
import json
from datetime import date, datetime, timedelta
import os

from typing import Optional, Dict, Union, List


def search_subtitles(results: int, movie_name: str, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Search for subtitles for any movie or TV series"
    results: number of results API should return, maximum 50
        movie_name: movie name and it can be as simple as "dark-knight" or it can contain a language (as full language or code, e.g. "english" or "en") and/or a year. You can also look for episodes of different TV series you want. Just make sure you call it like this: movie-name-separated-by-dashes
        
    """
    url = f"https://subsmax-subtitles.p.rapidapi.com/api/{results}/{movie_name}"
    querystring = {}
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "subsmax-subtitles.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def list_subtitles_by_id(limit: str, startid: int, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "List out subtitles in database"
    limit: limit of subtitles you can list at once (max 100)
        startid:  start ID from where you want to list subtitles.
        
    """
    url = f"https://subsmax-subtitles.p.rapidapi.com/api-list-subtitles/{startid}/{limit}"
    querystring = {}
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "subsmax-subtitles.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

