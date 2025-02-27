import requests
import json
from datetime import date, datetime, timedelta
import os

from typing import Optional, Dict, Union, List


def convert_to_itm(lat: int, long: int, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Convert WGS84 coordinates to ITM (Israeli Transverse Mercator) coordinates"
    
    """
    url = f"https://wgs84-to-itm.p.rapidapi.com/convert-to-itm"
    querystring = {'lat': lat, 'long': long, }
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "wgs84-to-itm.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

