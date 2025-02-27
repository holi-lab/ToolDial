import requests
import json
from datetime import date, datetime, timedelta
import os

from typing import Optional, Dict, Union, List


def get_taxi_fares(arr_lat: int, arr_lng: int, dep_lat: int, dep_lng: int, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Search fares by geo coordinates"
    arr_lat: Latitude of arrival point
        arr_lng: Longitude of arrival point
        dep_lat: Latitude of departure point
        dep_lng: Longitude of departure point
        
    """
    url = f"https://taxi-fare-calculator.p.rapidapi.com/search-geo"
    querystring = {'arr_lat': arr_lat, 'arr_lng': arr_lng, 'dep_lat': dep_lat, 'dep_lng': dep_lng, }
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "taxi-fare-calculator.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

