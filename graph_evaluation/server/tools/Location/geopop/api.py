import requests
import json
from datetime import date, datetime, timedelta
import os

from typing import Optional, Dict, Union, List


def get_buildings(lat: str, lon: str, radius: str, resource: str='all_buildings', toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Get square meters of built-up area (Europe only) inside the area defined by 
		
		- lat : latitude
		- lon : longitude
		- radius : radius (in meters) of the circular area around coordinates
		
		resource parameter allows to define which kind of  building resource is needed:
		- all_buildings : both residential and not residential buildings
		- residential: only residential buildings
		- not_residential: only not residential buildings
		- fileds: not built-up area"
    lat: latitude
        lon: longitude
        radius: radius
        resource: type of buildings
        
    """
    url = f"https://geopop.p.rapidapi.com/get_buildings"
    querystring = {'lat': lat, 'lon': lon, 'radius': radius, }
    if resource:
        querystring['resource'] = resource
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "geopop.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def get_population(lat: str, lon: str, radius: str, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Get number of resident population inside the area defined by 
		
		- lat : latitude
		- lon : longitude
		- radius : radius (in meters) of the circular area around coordinates"
    lat:  latitude (eg. 45.4643037)
        lon: longitude (e.g. 9.1907491)
        radius: radius in meters of circular area centered in (lat, lon) point
        
    """
    url = f"https://geopop.p.rapidapi.com/get_population"
    querystring = {'lat': lat, 'lon': lon, 'radius': radius, }
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "geopop.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

