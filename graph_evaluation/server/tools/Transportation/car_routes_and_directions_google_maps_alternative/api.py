import requests
import json
from datetime import date, datetime, timedelta
import os

from typing import Optional, Dict, Union, List


def getroute(end: str, start: str, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Get a basic route between two points for a car. Returned response is in GeoJSON format. This method does not accept any request body or parameters other than start coordinate and end coordinate."
    end: End point of route. Comma-separated longitude and latitude.
        start: Start point of route. Comma-separated longitude and latitude.
        
    """
    url = f"https://car-routes-and-directions-google-maps-alternative.p.rapidapi.com/directions/driving-car"
    querystring = {'end': end, 'start': start, }
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "car-routes-and-directions-google-maps-alternative.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

