import requests
import json
from datetime import date, datetime, timedelta
import os

from typing import Optional, Dict, Union, List


def water_vapour_data_by_coordinates(lat: int, lng: int, maxdistance: int=None, limit: int=None, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Get Water Vapour data by Coordinates"
    
    """
    url = f"https://ambee-water-vapor-data.p.rapidapi.com/waterVapor/latest/by-lat-lng"
    querystring = {'lat': lat, 'lng': lng, }
    if maxdistance:
        querystring['maxDistance'] = maxdistance
    if limit:
        querystring['limit'] = limit
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "ambee-water-vapor-data.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def water_vapour_history_by_lat_lng(lat: int, lng: int, enddate: str, startdate: str, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Water vapour History by lat lng"
    
    """
    url = f"https://ambee-water-vapor-data.p.rapidapi.com/waterVapor/history/by-lat-lng"
    querystring = {'lat': lat, 'lng': lng, 'endDate': enddate, 'startDate': startdate, }
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "ambee-water-vapor-data.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

