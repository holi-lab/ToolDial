import requests
import json
from datetime import date, datetime, timedelta
import os

from typing import Optional, Dict, Union, List


def get_all_film_locations(tr_latitude: int, tr_longitude: int, bl_longitude: int, bl_latitude: int, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "All of the info + required boundaries params"
    
    """
    url = f"https://filming-locations1.p.rapidapi.com/filmlocations"
    querystring = {'tr_latitude': tr_latitude, 'tr_longitude': tr_longitude, 'bl_longitude': bl_longitude, 'bl_latitude': bl_latitude, }
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "filming-locations1.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

