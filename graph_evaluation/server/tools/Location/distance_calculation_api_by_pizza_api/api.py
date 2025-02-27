import requests
import json
from datetime import date, datetime, timedelta
import os

from typing import Optional, Dict, Union, List


def calculate_distance_by_lat_long(metric: str, lat2: str, lon2: str, lon1: str, lat1: str, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Calculate Distance between to place by their latitude and longitudes and metric you want the output of."
    
    """
    url = f"https://distance-calculation-api-by-pizza-api.p.rapidapi.com/distance"
    querystring = {'metric': metric, 'lat2': lat2, 'lon2': lon2, 'lon1': lon1, 'lat1': lat1, }
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "distance-calculation-api-by-pizza-api.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

