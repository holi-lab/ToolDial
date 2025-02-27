import requests
import json
from datetime import date, datetime, timedelta
import os

from typing import Optional, Dict, Union, List


def search(order_by: str=None, limit: int=None, dec: int=None, offset: int=None, match_type: str=None, ra: int=None, exact: str=None, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Search endpoint can be used to get information for stars and deep space objects."
    
    """
    url = f"https://astronomy.p.rapidapi.com/api/v2/search"
    querystring = {}
    if order_by:
        querystring['order_by'] = order_by
    if limit:
        querystring['limit'] = limit
    if dec:
        querystring['dec'] = dec
    if offset:
        querystring['offset'] = offset
    if match_type:
        querystring['match_type'] = match_type
    if ra:
        querystring['ra'] = ra
    if exact:
        querystring['exact'] = exact
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "astronomy.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def get_positions_for_body(to_date: str, body: str, latitude: int, from_date: str, longitude: int, time: str='12:00:00', elevation: int=166, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Returns properties of the given body for the given date range in tabular format.
		
		Supported bodies are "sun" ,"moon", "mercury", "venus", "earth", "mars", "jupiter", "saturn", "uranus", "neptune", "pluto""
    
    """
    url = f"https://astronomy.p.rapidapi.com/api/v2/bodies/positions/{body}"
    querystring = {'to_date': to_date, 'latitude': latitude, 'from_date': from_date, 'longitude': longitude, }
    if time:
        querystring['time'] = time
    if elevation:
        querystring['elevation'] = elevation
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "astronomy.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def get_all_bodies_positions(latitude: int, longitude: int, from_date: str, to_date: str, elevation: int=166, time: str='12:00:00', toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Returns a iterable list of bodies and their properties in tabular format."
    
    """
    url = f"https://astronomy.p.rapidapi.com/api/v2/bodies/positions"
    querystring = {'latitude': latitude, 'longitude': longitude, 'from_date': from_date, 'to_date': to_date, }
    if elevation:
        querystring['elevation'] = elevation
    if time:
        querystring['time'] = time
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "astronomy.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

