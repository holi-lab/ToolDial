import requests
import json
from datetime import date, datetime, timedelta
import os

from typing import Optional, Dict, Union, List


def get_heightmap_from_two_geopoints(lng0: int, lng1: int, lat1: int, lat0: int, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "you provide two geopoints( latitude, longitude). Server returns image/png with heightmap within this borders."
    
    """
    url = f"https://heightmap-from-latitude-and-longitude.p.rapidapi.com/"
    querystring = {'lng0': lng0, 'lng1': lng1, 'lat1': lat1, 'lat0': lat0, }
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "heightmap-from-latitude-and-longitude.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

