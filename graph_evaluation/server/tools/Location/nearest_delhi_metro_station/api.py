import requests
import json
from datetime import date, datetime, timedelta
import os

from typing import Optional, Dict, Union, List


def nearest_metro_station(long: str, lat: str, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "This endpoint is a **GET** method API that returns -
		
		- **Station name**
		- **Latitude** & **Longitude** of the nearest Delhi Metro station
		- **Google Maps direction**"
    
    """
    url = f"https://nearest-delhi-metro-station.p.rapidapi.com/nearestmetro"
    querystring = {'long': long, 'lat': lat, }
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "nearest-delhi-metro-station.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

