import requests
import json
from datetime import date, datetime, timedelta
import os

from typing import Optional, Dict, Union, List


def get_sunrise_and_sunset_times(latitude: int, date: str, longitude: int, timezoneid: str='America/New_York', toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Get sunrise and sunset times by date, latitude, and longitude."
    
    """
    url = f"https://sunrise-sunset-times.p.rapidapi.com/getSunriseAndSunset"
    querystring = {'latitude': latitude, 'date': date, 'longitude': longitude, }
    if timezoneid:
        querystring['timeZoneId'] = timezoneid
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "sunrise-sunset-times.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

