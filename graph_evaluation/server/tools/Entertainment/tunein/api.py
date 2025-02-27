import requests
import json
from datetime import date, datetime, timedelta
import os

from typing import Optional, Dict, Union, List


def broadcaster_endpoint(title: str, artist: str, album: str, commericial: bool, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Your title and artist information should be set to be updated by your broadcast software in correspondence with the currently playing song."
    title: The name of the song
        artist: The artist associated with the song
        album: The name of the associated album
        commericial: Set to true if the now playing broadcast is in commercial rather than a song
        
    """
    url = f"https://community-tunein.p.rapidapi.com/Playing.ashx/"
    querystring = {'title': title, 'artist': artist, 'album': album, 'commericial': commericial, }
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "community-tunein.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

