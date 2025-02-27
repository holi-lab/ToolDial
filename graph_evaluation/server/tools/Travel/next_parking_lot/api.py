import requests
import json
from datetime import date, datetime, timedelta
import os

from typing import Optional, Dict, Union, List


def parking_lot_details(is_id: int, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Getting detailed information about a given parking lot based on their id."
    
    """
    url = f"https://next-parking-lot.p.rapidapi.com/location/tags/{is_id}"
    querystring = {}
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "next-parking-lot.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def parking_lot_collection(lngmin: int, densitylevel: int, latmin: int, lngmax: int, latmax: int, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Getting parking lots within a given bounding box.
		
		DensityLevel must be between 3 and 15.
		
		There is a maximum of 2000 parking lots per request."
    
    """
    url = f"https://next-parking-lot.p.rapidapi.com/location/{latmin}/{latmax}/{lngmin}/{lngmax}/{densitylevel}"
    querystring = {}
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "next-parking-lot.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

