import requests
import json
from datetime import date, datetime, timedelta
import os

from typing import Optional, Dict, Union, List


def get_stations_within_10_km(region: str, longitude: int, latitude: int, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Return nearest charging stations within 10 km radius."
    region: us=United States, ca=Canada, uk=United Kingdom, nz=New Zealand, hk=Hong Kong
        
    """
    url = f"https://ev-charging-stations.p.rapidapi.com/get_stations_10km"
    querystring = {'region': region, 'longitude': longitude, 'latitude': latitude, }
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "ev-charging-stations.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def get_stations_within_5_km(region: str, longitude: int, latitude: int, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Return nearest charging stations within 5 km radius."
    region: us=United States, ca=Canada, uk=United Kingdom, nz=New Zealand, hk=Hong Kong
        
    """
    url = f"https://ev-charging-stations.p.rapidapi.com/get_stations_5km"
    querystring = {'region': region, 'longitude': longitude, 'latitude': latitude, }
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "ev-charging-stations.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def get_stations_within_1_km(region: str, latitude: int, longitude: int, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Return nearest charging stations within 1 km radius."
    region: us=United States, ca=Canada, uk=United Kingdom, nz=New Zealand, hk=Hong Kong
        
    """
    url = f"https://ev-charging-stations.p.rapidapi.com/get_stations_1km"
    querystring = {'region': region, 'latitude': latitude, 'longitude': longitude, }
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "ev-charging-stations.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

