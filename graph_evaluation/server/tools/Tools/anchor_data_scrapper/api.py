import requests
import json
from datetime import date, datetime, timedelta
import os

from typing import Optional, Dict, Union, List


def get_podcast_episodes(username: str, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Get Podcast episodes with Audio, image, description using the username of the podcast."
    username: The username of the anchor podcase user for example:
https://anchor.fm/**financialfreedomtribe**
        
    """
    url = f"https://anchor-data-scrapper.p.rapidapi.com/episodes/{username}"
    querystring = {}
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "anchor-data-scrapper.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

