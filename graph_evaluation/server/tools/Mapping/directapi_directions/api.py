import requests
import json
from datetime import date, datetime, timedelta
import os

from typing import Optional, Dict, Union, List


def autocomplete(q: str, radius: str='5000', limit: str='5', location: str='-52,113', toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Provides address suggestions as the user types"
    radius: Radius in meters around the location
        
    """
    url = f"https://directapi-directions.p.rapidapi.com/autocomplete"
    querystring = {'q': q, }
    if radius:
        querystring['radius'] = radius
    if limit:
        querystring['limit'] = limit
    if location:
        querystring['location'] = location
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "directapi-directions.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def search(q: str, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Search for locations based on a text search"
    
    """
    url = f"https://directapi-directions.p.rapidapi.com/search"
    querystring = {'q': q, }
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "directapi-directions.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def directions(stops: str, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Get an optimized route through a set of 2 or more stops"
    stops: A list of pipe(|) separated stops. Each stop can be an address, a set of coordinates, or a Google place_id. You can specify an address as the starting or ending point of the trip by adding \\\\\\\\\\\\\\\"start:\\\\\\\\\\\\\\\" or \\\\\\\\\\\\\\\"end:\\\\\\\\\\\\\\\" before the address.
        
    """
    url = f"https://directapi-directions.p.rapidapi.com/directions"
    querystring = {'stops': stops, }
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "directapi-directions.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

