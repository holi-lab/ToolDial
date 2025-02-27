import requests
import json
from datetime import date, datetime, timedelta
import os

from typing import Optional, Dict, Union, List


def smart_charging(lon: int, lat: int, start_time: str, end_time: str, minutes: int, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Generates an optimized electric vehicle charging schedule"
    
    """
    url = f"https://smart-electric-vehicle-charging.p.rapidapi.com/smart_charging"
    querystring = {'lon': lon, 'lat': lat, 'start_time': start_time, 'end_time': end_time, 'minutes': minutes, }
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "smart-electric-vehicle-charging.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

