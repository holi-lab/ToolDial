import requests
import json
from datetime import date, datetime, timedelta
import os

from typing import Optional, Dict, Union, List


def one_to_one(lat2: int, long1: int, lat1: int, long2: int, unit: str='miles', toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Distance between two locations."
    unit: Default: kilometers. 
Supported: kilometers, meters, miles, nautical_miles, feet, inches, 
        
    """
    url = f"https://lat-long-distance-calculator.p.rapidapi.com/dist"
    querystring = {'lat2': lat2, 'long1': long1, 'lat1': lat1, 'long2': long2, }
    if unit:
        querystring['unit'] = unit
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "lat-long-distance-calculator.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

