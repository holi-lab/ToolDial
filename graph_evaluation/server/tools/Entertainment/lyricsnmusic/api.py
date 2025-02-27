import requests
import json
from datetime import date, datetime, timedelta
import os

from typing import Optional, Dict, Union, List


def search_lyrics(q: str='coldplay clocks', artist: str=None, track: str=None, lyrics: str=None, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "By artist name, song title, or lyrics words"
    
    """
    url = f"https://community-lyricsnmusic.p.rapidapi.com/songs"
    querystring = {}
    if q:
        querystring['q'] = q
    if artist:
        querystring['artist'] = artist
    if track:
        querystring['track'] = track
    if lyrics:
        querystring['lyrics'] = lyrics
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "community-lyricsnmusic.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

