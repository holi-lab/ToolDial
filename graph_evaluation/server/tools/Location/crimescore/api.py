import requests
import json
from datetime import date, datetime, timedelta
import os

from typing import Optional, Dict, Union, List


def crimescore(lat: int, lon: int, f: str='json', is_id: int=174, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Get CrimeScore rating data at your current location"
    lat: Latitude of Location
        lon: Longitude of Location
        f: Format of resulting output. json, csv. JSON is default.
        is_id: Data set ID of crime data set from YourMapper (if empty it will auto detect)
        
    """
    url = f"https://crimescore.p.rapidapi.com/crimescore"
    querystring = {'lat': lat, 'lon': lon, }
    if f:
        querystring['f'] = f
    if is_id:
        querystring['id'] = is_id
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "crimescore.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

