import requests
import json
from datetime import date, datetime, timedelta
import os

from typing import Optional, Dict, Union, List


def get_distance(end_lat: int, unit: str, end_lng: int, start_lng: int, start_lat: int, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Calculate the distance between two GPS co-ordinate points."
    
    """
    url = f"https://distance-calculator1.p.rapidapi.com/v1/getdistance"
    querystring = {'end_lat': end_lat, 'unit': unit, 'end_lng': end_lng, 'start_lng': start_lng, 'start_lat': start_lat, }
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "distance-calculator1.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

