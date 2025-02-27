import requests
import json
from datetime import date, datetime, timedelta
import os

from typing import Optional, Dict, Union, List


def gettimezone(longitude: int, latitude: int, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Finds the local timezone for any given geo-location point by lat and long and returns timezone information with Timezone name, Timezone id and current local time."
    
    """
    url = f"https://geocodeapi.p.rapidapi.com/GetTimezone"
    querystring = {'longitude': longitude, 'latitude': latitude, }
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "geocodeapi.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

