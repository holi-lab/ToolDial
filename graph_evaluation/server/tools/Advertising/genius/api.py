import requests
import json
from datetime import date, datetime, timedelta
import os

from typing import Optional, Dict, Union, List


def artists(authorization: str=None, artist: str=None, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "artist"
    
    """
    url = f"https://genius7.p.rapidapi.com/artist"
    querystring = {}
    if authorization:
        querystring['Authorization'] = authorization
    if artist:
        querystring['artist'] = artist
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "genius7.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def albums(authorization: str=None, albums: str=None, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "albums"
    
    """
    url = f"https://genius7.p.rapidapi.com/albums"
    querystring = {}
    if authorization:
        querystring['Authorization'] = authorization
    if albums:
        querystring['albums'] = albums
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "genius7.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

