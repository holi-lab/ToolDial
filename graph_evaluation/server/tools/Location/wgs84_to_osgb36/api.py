import requests
import json
from datetime import date, datetime, timedelta
import os

from typing import Optional, Dict, Union, List


def wgs84_to_osgb36(longitude: int, latitude: int, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Convert a WGS84 latitude and longitude to OSGB36 easting and northing (British National Grid)."
    
    """
    url = f"https://wgs84-to-osgb36.p.rapidapi.com/wgs84ToOsgb36/{latitude}/{longitude}"
    querystring = {}
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "wgs84-to-osgb36.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

