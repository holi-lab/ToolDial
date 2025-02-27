import requests
import json
from datetime import date, datetime, timedelta
import os

from typing import Optional, Dict, Union, List


def pharmacity_products(query: str='query GetProducts{product{id,name}}', toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Get Pharmacity's products selling online"
    
    """
    url = f"https://pharmacity.p.rapidapi.com/graphqli"
    querystring = {}
    if query:
        querystring['query'] = query
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "pharmacity.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def pharmacity_stores(longitude: str='106.6963943', latitude: str='10.8136517', distance: str='2000', query: str='query getstorebydistance ($longitude: Float, $latitude: Float, $distance: Float){ store(longitude: $longitude, latitude: $latitude, distance: $distance){ id, db_id, name, address } }', toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "List of Pharmacity stores in Vietnam"
    
    """
    url = f"https://pharmacity.p.rapidapi.com/graphql"
    querystring = {}
    if longitude:
        querystring['longitude'] = longitude
    if latitude:
        querystring['latitude'] = latitude
    if distance:
        querystring['distance'] = distance
    if query:
        querystring['query'] = query
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "pharmacity.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

