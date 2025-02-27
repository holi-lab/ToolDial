import requests
import json
from datetime import date, datetime, timedelta
import os

from typing import Optional, Dict, Union, List


def latitude_and_longitude(toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Speed Limit Cameras data worldwide by longitude and latitude"
    
    """
    url = f"https://speed-limit-cameras.p.rapidapi.com/?lat_gte=52&lat_lte=55&lon_gte=22&lon_lte=25"
    querystring = {}
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "speed-limit-cameras.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

