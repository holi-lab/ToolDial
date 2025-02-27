import requests
import json
from datetime import date, datetime, timedelta
import os

from typing import Optional, Dict, Union, List


def sunposition(lat: str, lon: str, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Parameters:
		lat (float): The latitude of the location for which you want to get the solar position. The value should be between -90 and 90 degrees.
		lon (float): The longitude of the location for which you want to get the solar position. The value should be between -180 and 180 degrees."
    
    """
    url = f"https://sun-seeker-api.p.rapidapi.com/sunposition"
    querystring = {'lat': lat, 'lon': lon, }
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "sun-seeker-api.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

