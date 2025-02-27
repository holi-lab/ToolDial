import requests
import json
from datetime import date, datetime, timedelta
import os

from typing import Optional, Dict, Union, List


def xyz_map_tiles_for_ocean_conditions(variable: str, datetime: str, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Wavebase Global Ocean Tiles API:
		
		Wave Signficant Height (GET): https://api.wavebase.app/v1/tiles/singleband/{variable}/{date-timeslot - YYYYMMDDHH}/0/0/0.png
		
		e.g.: https://api.wavebase.app/v1/tiles/singleband/VHM0/2023051703/0/0/0.png 
		
		Currently only Tile at 0/0/0 is available Free.  Contact:  info@wavebase.app for additional tiles."
    
    """
    url = f"https://wavebase.p.rapidapi.com/v1/tiles/singleband/{variable}/{datetime}/0/0/0.png"
    querystring = {}
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "wavebase.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def ocean_conditions_closest_to_latitude_longitude(latitude: int, longitude: int, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "​https://api.wavebase.app/v4/conditions/search/{latitude}/{longitude}
		
		Search for current and predicted ocean conditions closest to a given latitude/longitude.
		
		Returned conditions include the closest point found, along with Waves, Tides and Weather conditions are the location."
    
    """
    url = f"https://wavebase.p.rapidapi.com/v4/conditions/search/{latitude}/{longitude}"
    querystring = {}
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "wavebase.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

