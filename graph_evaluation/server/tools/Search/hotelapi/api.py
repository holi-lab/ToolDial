import requests
import json
from datetime import date, datetime, timedelta
import os

from typing import Optional, Dict, Union, List


def hotelsearch(hoteldata: str='HotelTest', toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Search Hotel List"
    hoteldata: Hotel Result
        
    """
    url = f"https://hotelapi.p.rapidapi.comHotelSearch"
    querystring = {}
    if hoteldata:
        querystring['HotelData'] = hoteldata
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "hotelapi.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

