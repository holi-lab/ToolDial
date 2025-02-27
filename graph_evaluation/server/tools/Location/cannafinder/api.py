import requests
import json
from datetime import date, datetime, timedelta
import os

from typing import Optional, Dict, Union, List


def shops_in_boundingbox_delta(longitudedelta: int, latitudedelta: int, lon: int, lat: int, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Find shops in boundingbox with delta"
    
    """
    url = f"https://cannafinder.p.rapidapi.com/shops-in-boundingbox-delta"
    querystring = {'longitudedelta': longitudedelta, 'latitudedelta': latitudedelta, 'lon': lon, 'lat': lat, }
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "cannafinder.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def shops_in_boundingbox(maxlon: int, minlat: int, maxlat: int, minlon: int, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Find shops in a coordinates boundingbox"
    
    """
    url = f"https://cannafinder.p.rapidapi.com/shops-in-boundingbox"
    querystring = {'maxLon': maxlon, 'minLat': minlat, 'maxLat': maxlat, 'minLon': minlon, }
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "cannafinder.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def shops_in_radius_around_coordinates(radius: int, lat: int, lon: int, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Find shops in input radius and coordinates. Returns multiple tags for every entry and distance in km."
    radius: in meter. 
        
    """
    url = f"https://cannafinder.p.rapidapi.com/shops-in-radius-around-coordinates"
    querystring = {'radius': radius, 'lat': lat, 'lon': lon, }
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "cannafinder.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

