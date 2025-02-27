import requests
import json
from datetime import date, datetime, timedelta
import os

from typing import Optional, Dict, Union, List


def your_vinyl_store(year: str=None, artist: str=None, genre: str=None, album: str=None, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Best vinyl records in your collection"
    
    """
    url = f"https://yourvinylstore.p.rapidapi.com/endpoint"
    querystring = {}
    if year:
        querystring['Year'] = year
    if artist:
        querystring['Artist'] = artist
    if genre:
        querystring['Genre'] = genre
    if album:
        querystring['Album'] = album
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "yourvinylstore.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def your_vinyl_store_copy(year: str=None, artist: str=None, genre: str=None, album: str=None, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Best vinyl records in your collection"
    
    """
    url = f"https://yourvinylstore.p.rapidapi.com/endpoint"
    querystring = {}
    if year:
        querystring['Year'] = year
    if artist:
        querystring['Artist'] = artist
    if genre:
        querystring['Genre'] = genre
    if album:
        querystring['Album'] = album
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "yourvinylstore.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

