import requests
import json
from datetime import date, datetime, timedelta
import os

from typing import Optional, Dict, Union, List


def tides(longitude: int, latitude: int, interval: int=60, radius: int=None, timestamp: int=None, duration: str='1440', toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Endpoint for getting Tide predictions like extremes and water level"
    longitude: Longitude in range from -180 to 180
        latitude: Latitude in range from -90 to 90
        interval: Interval means number of minutes between the returned measurements. Please note that one response can contain max 10 080 predicted heights, so duration/interval can't be bigger than 10 080.
        radius: When no prediction is found in requested coordinates, API tries to return the nearest prediction. You can limit the radius by setting radius parameter to any positive integer.
        timestamp: Timestamp (number of seconds since the unix epoch) of the prediction beginning. Datetime represented with this timestamp will be floored to whole minute. Defaults to current timestamp
        duration: Duration is the number of minutes for which the forecast should be calculated. Default and minimum is 1 440 (one day). Please note that one response can contain max 10 080 predicted heights, so duration/interval can't be bigger than 10 080.
        
    """
    url = f"https://tides.p.rapidapi.com/tides"
    querystring = {'longitude': longitude, 'latitude': latitude, }
    if interval:
        querystring['interval'] = interval
    if radius:
        querystring['radius'] = radius
    if timestamp:
        querystring['timestamp'] = timestamp
    if duration:
        querystring['duration'] = duration
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "tides.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

