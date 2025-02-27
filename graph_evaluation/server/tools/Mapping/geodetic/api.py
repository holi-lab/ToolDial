import requests
import json
from datetime import date, datetime, timedelta
import os

from typing import Optional, Dict, Union, List


def azimuth(direction: str, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Calculates the corresponding azimuth using a 32-wind compass rose.
		A direction of N equals an azimuth of 0 and a direction of S equals an azimuth of 180."
    
    """
    url = f"https://geodetic.p.rapidapi.com/azimuth"
    querystring = {'direction': direction, }
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "geodetic.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

