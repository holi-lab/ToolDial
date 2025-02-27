import requests
import json
from datetime import date, datetime, timedelta
import os

from typing import Optional, Dict, Union, List


def convert_coordinates_to_timezone_string(lng: str, lat: str, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "This endpoint converts coordinates(latitude,longitude) to a timezone string which can be used for time calculations."
    lng: Longitude
        lat: Latitude
        
    """
    url = f"https://location-to-timezone.p.rapidapi.com/v1/point-to-timezone"
    querystring = {'lng': lng, 'lat': lat, }
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "location-to-timezone.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

