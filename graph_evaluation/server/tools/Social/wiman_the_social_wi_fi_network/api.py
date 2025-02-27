import requests
import json
from datetime import date, datetime, timedelta
import os

from typing import Optional, Dict, Union, List


def venue_search(latitude: str, longitude: str, radius: int=1, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    ""
    latitude: latitude
        longitude: longitude
        radius: radius between 1 and 6
        
    """
    url = f"https://wiman-wiman---the-social-wi-fi-network.p.rapidapi.com/venue/search/"
    querystring = {'latitude': latitude, 'longitude': longitude, }
    if radius:
        querystring['radius'] = radius
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "wiman-wiman---the-social-wi-fi-network.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

