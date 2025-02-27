import requests
import json
from datetime import date, datetime, timedelta
import os

from typing import Optional, Dict, Union, List


def is_a_coordinate_water(longitude: int, latitude: int, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Given a latitude and longitude, return if that coordinate on Earth is water."
    
    """
    url = f"https://isitwater-com.p.rapidapi.com/"
    querystring = {'longitude': longitude, 'latitude': latitude, }
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "isitwater-com.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

