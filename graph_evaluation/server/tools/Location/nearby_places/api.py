import requests
import json
from datetime import date, datetime, timedelta
import os

from typing import Optional, Dict, Union, List


def nearby_arcgis(lng: int, lat: int, radius: int=500, type: str='coffee shop', toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Returns nearby places sorted by distance from the origin coordinates in ascending order.
		
		Only type values from **/v2/types (ArcGIS)** are valid.
		
		Maximum returned places is 50."
    lng: Longitude
        lat: Latitude
        radius: **Meters**
Default: 200
Maximum: 50000
        type: Default: 'coffee shop'
        
    """
    url = f"https://nearby-places.p.rapidapi.com/v2/nearby"
    querystring = {'lng': lng, 'lat': lat, }
    if radius:
        querystring['radius'] = radius
    if type:
        querystring['type'] = type
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "nearby-places.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def nearby_google(lat: int, lng: int, type: str='cafe', radius: int=200, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Returns a **JSON array of nearby places** sorted by distance from the origin coordinates in ascending order.
		
		Only values from **/types (Google)** are valid types.
		
		Maximum places returned per request is 20."
    lat: Latitude
        lng: Longitude
        type: Default is **cafe**.
        radius: **Meters**
Default: 200
Max: 50000
        
    """
    url = f"https://nearby-places.p.rapidapi.com/nearby"
    querystring = {'lat': lat, 'lng': lng, }
    if type:
        querystring['type'] = type
    if radius:
        querystring['radius'] = radius
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "nearby-places.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def types_google(toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Get a list of establishment types.
		
		These types will only work with **/nearby (Google)**."
    
    """
    url = f"https://nearby-places.p.rapidapi.com/types"
    querystring = {}
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "nearby-places.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def random_google(lat: int, lng: int, type: str='cafe', radius: int=500, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Return a random place nearby."
    lat: Latitude
        lng: Longitude
        type: Default: 'cafe'
        radius: **Meters**
Default: 200
Max: 50000

        
    """
    url = f"https://nearby-places.p.rapidapi.com/random"
    querystring = {'lat': lat, 'lng': lng, }
    if type:
        querystring['type'] = type
    if radius:
        querystring['radius'] = radius
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "nearby-places.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def place_details_google(is_id: str, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Get contact and opening hours for a place.
		
		Requires a **place id** provided by **/nearby (Google)**."
    id: Place ID.
This value can be retrieved from */nearby* or */random* endpoints.
        
    """
    url = f"https://nearby-places.p.rapidapi.com/details"
    querystring = {'id': is_id, }
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "nearby-places.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def types_arcgis(toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Types (categories) of establishments. These types will only work with **/v2/nearby (ArcGIS)**."
    
    """
    url = f"https://nearby-places.p.rapidapi.com/v2/types"
    querystring = {}
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "nearby-places.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

