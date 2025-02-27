import requests
import json
from datetime import date, datetime, timedelta
import os

from typing import Optional, Dict, Union, List


def get_league_of_legends_character(toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "This endpoint will return back a random league of legends character with their descriptions including: 
		- title
		- age
		- region of origin
		- primary lane
		- release date
		- gender
		- race"
    
    """
    url = f"https://league-of-legends-character-generator.p.rapidapi.com/generate"
    querystring = {}
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "league-of-legends-character-generator.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

