import requests
import json
from datetime import date, datetime, timedelta
import os

from typing import Optional, Dict, Union, List


def soil_history_data(lng: int, lat: int, startdate: str, enddate: str, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Ambee  Soil History Data"
    
    """
    url = f"https://ambee-soil-data.p.rapidapi.com/soil/history/by-lat-lng"
    querystring = {'lng': lng, 'lat': lat, 'startDate': startdate, 'endDate': enddate, }
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "ambee-soil-data.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def soil_data_by_lat_lng(lng: int, lat: int, pollutants: str=None, limit: int=None, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "get nearest places readings based around the given lattitude and logitude"
    
    """
    url = f"https://ambee-soil-data.p.rapidapi.com/soil/latest/by-lat-lng"
    querystring = {'lng': lng, 'lat': lat, }
    if pollutants:
        querystring['pollutants'] = pollutants
    if limit:
        querystring['limit'] = limit
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "ambee-soil-data.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

