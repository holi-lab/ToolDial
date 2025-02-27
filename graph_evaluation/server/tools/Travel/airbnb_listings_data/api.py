import requests
import json
from datetime import date, datetime, timedelta
import os

from typing import Optional, Dict, Union, List


def airbnblistingsdata(selat: int, selng: int, nwlng: int, nwlat: int, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "[To be provided by Jae]"
    
    """
    url = f"https://airbnb-listings-data.p.rapidapi.com/getListingsData"
    querystring = {'seLat': selat, 'seLng': selng, 'nwLng': nwlng, 'nwLat': nwlat, }
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "airbnb-listings-data.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

