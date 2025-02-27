import requests
import json
from datetime import date, datetime, timedelta
import os

from typing import Optional, Dict, Union, List


def search_by_url(spotify_url: str, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Given a **Spotify url**, this returns details about *tracks*, *albums* and *artists*."
    
    """
    url = f"https://musiclinkssapi.p.rapidapi.com/search/url"
    querystring = {'spotify_url': spotify_url, }
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "musiclinkssapi.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def get_artist_s_albums(spotify_url: str, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Given a **Spotify url**,  returns basic details about albums of artist."
    
    """
    url = f"https://musiclinkssapi.p.rapidapi.com/artist_albums"
    querystring = {'spotify_url': spotify_url, }
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "musiclinkssapi.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def search_by_query(q: str, type: str, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Given a **query** and **type** of element, this returns details about *tracks*, *albums* and *artists*."
    
    """
    url = f"https://musiclinkssapi.p.rapidapi.com/search/query"
    querystring = {'q': q, 'type': type, }
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "musiclinkssapi.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

