import requests
import json
from datetime import date, datetime, timedelta
import os

from typing import Optional, Dict, Union, List


def timezone(lng: int, timestamp: int, lat: int, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Get the timezone of the coordinate location."
    lng: Longitude
        timestamp: The desired time as **seconds **since midnight, January 1, 1970 UTC.
        lat: Latitude
        
    """
    url = f"https://timezone-by-coordinates.p.rapidapi.com/timezone"
    querystring = {'lng': lng, 'timestamp': timestamp, 'lat': lat, }
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "timezone-by-coordinates.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

