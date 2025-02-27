import requests
import json
from datetime import date, datetime, timedelta
import os

from typing import Optional, Dict, Union, List


def get_track_streams_playback_count(isrc: str='CA5KR1821202', spotify_track_id: str='6ho0GyrWZN3mhi9zVRW7xi', toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Get the stream number / play count of any song on Spotify!"
    
    """
    url = f"https://spotify-track-streams-playback-count1.p.rapidapi.com/tracks/spotify_track_streams"
    querystring = {}
    if isrc:
        querystring['isrc'] = isrc
    if spotify_track_id:
        querystring['spotify_track_id'] = spotify_track_id
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "spotify-track-streams-playback-count1.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

