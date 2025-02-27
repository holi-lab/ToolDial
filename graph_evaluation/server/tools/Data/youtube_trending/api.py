import requests
import json
from datetime import date, datetime, timedelta
import os

from typing import Optional, Dict, Union, List


def trending(type: str='gaming', country: str='US', toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Gather Trending Videos from Youtube"
    type: enum for video type:
- default
- music
- gaming
- movies
        country: Country code to retrieve trending videos for the provided country.
E.g, US, DE, JP etc.
        
    """
    url = f"https://youtube-trending.p.rapidapi.com/trending"
    querystring = {}
    if type:
        querystring['type'] = type
    if country:
        querystring['country'] = country
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "youtube-trending.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

