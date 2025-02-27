import requests
import json
from datetime import date, datetime, timedelta
import os

from typing import Optional, Dict, Union, List


def incidents(date: str, longitude: int, latitude: int, radius: str, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "This API returns verified high-risk events. User is required to specify all parameters including Longitude, Latitude, Date (mm-dd-yyyy) and Radius (up to 100 Km)."
    
    """
    url = f"https://incident-factor.p.rapidapi.com/rapidApi/getIncidents"
    querystring = {'date': date, 'longitude': longitude, 'latitude': latitude, 'radius': radius, }
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "incident-factor.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def location_risk(latitude: int, longitude: int, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Get zone incident factor with latitude and longitude"
    
    """
    url = f"https://incident-factor.p.rapidapi.com/rapidApi/zone/incidentFactor"
    querystring = {'latitude': latitude, 'longitude': longitude, }
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "incident-factor.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

