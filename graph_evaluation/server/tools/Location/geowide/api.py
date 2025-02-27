import requests
import json
from datetime import date, datetime, timedelta
import os

from typing import Optional, Dict, Union, List


def distance(start_longitude: int, start_latitude: int, end_latitude: int, end_longitude: int, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "This API endpoint takes latitude and longitude coordinates for two points and calculates the geodesic and great circle distances between them. It returns a json object with formatted coordinate information for each point, along with the distances in miles, kilometres, meters, feet, and nautical miles for both geodesic and great circle measurements."
    
    """
    url = f"https://geowide.p.rapidapi.com/distance"
    querystring = {'start_longitude': start_longitude, 'start_latitude': start_latitude, 'end_latitude': end_latitude, 'end_longitude': end_longitude, }
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "geowide.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

