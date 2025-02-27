import requests
import json
from datetime import date, datetime, timedelta
import os

from typing import Optional, Dict, Union, List


def parameters(long1: int, lat2: int, long2: int, lat1: int, unit: str='K', toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Get Parameters"
    long1: Longitude of Point 1
        lat2: Latitude of Point 2
        long2: Longitude of Point 2
        lat1: Latitude of Point 1
        unit: Distance Unit, Default: K (K: Kilometar, M: Miles, N: Nautical Miles)
        
    """
    url = f"https://010pixel-distance-v1.p.rapidapi.com/"
    querystring = {'long1': long1, 'lat2': lat2, 'long2': long2, 'lat1': lat1, }
    if unit:
        querystring['unit'] = unit
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "010pixel-distance-v1.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

