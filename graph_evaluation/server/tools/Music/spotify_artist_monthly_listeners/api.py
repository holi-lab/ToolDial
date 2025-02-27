import requests
import json
from datetime import date, datetime, timedelta
import os

from typing import Optional, Dict, Union, List


def get_artist_monthly_listeners_count(spotify_artist_id: str, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Get the monthly listeners count of any artist on Spotify!"
    
    """
    url = f"https://spotify-artist-monthly-listeners.p.rapidapi.com/artists/spotify_artist_monthly_listeners"
    querystring = {'spotify_artist_id': spotify_artist_id, }
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "spotify-artist-monthly-listeners.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

