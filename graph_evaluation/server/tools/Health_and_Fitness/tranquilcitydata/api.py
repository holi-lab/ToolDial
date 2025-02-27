import requests
import json
from datetime import date, datetime, timedelta
import os

from typing import Optional, Dict, Union, List


def bristol_grid_tranquil(lng: str, lat: str, geo: str=None, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "All"
    
    """
    url = f"https://tranquilcitydata.p.rapidapi.com/v1/bristol/grid/tranquil/"
    querystring = {'lng': lng, 'lat': lat, }
    if geo:
        querystring['geo'] = geo
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "tranquilcitydata.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def bristol_streets_tranquil(lat: str, lng: str, geo: str=None, osm_id: str=None, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "All"
    
    """
    url = f"https://tranquilcitydata.p.rapidapi.com/v1/bristol/streets/tranquil/"
    querystring = {'lat': lat, 'lng': lng, }
    if geo:
        querystring['geo'] = geo
    if osm_id:
        querystring['osm_id'] = osm_id
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "tranquilcitydata.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def london_streets_tranquil(lng: str='-0.1', osm_id: str=None, lat: str='51.5', geo: str=None, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "All"
    
    """
    url = f"https://tranquilcitydata.p.rapidapi.com/v1/london/streets/tranquil/"
    querystring = {}
    if lng:
        querystring['lng'] = lng
    if osm_id:
        querystring['osm_id'] = osm_id
    if lat:
        querystring['lat'] = lat
    if geo:
        querystring['geo'] = geo
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "tranquilcitydata.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def london_streets_greeninfrastructure(lng: str='-0.1', osm_id: str=None, geo: str=None, lat: str='51.5', toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "All"
    
    """
    url = f"https://tranquilcitydata.p.rapidapi.com/v1/london/streets/greeninfrastructure/"
    querystring = {}
    if lng:
        querystring['lng'] = lng
    if osm_id:
        querystring['osm_id'] = osm_id
    if geo:
        querystring['geo'] = geo
    if lat:
        querystring['lat'] = lat
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "tranquilcitydata.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def london_streets_environmental(osm_id: str=None, lng: str='-0.1', lat: str='51.5', geo: str=None, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "All"
    
    """
    url = f"https://tranquilcitydata.p.rapidapi.com/v1/london/streets/environmental/"
    querystring = {}
    if osm_id:
        querystring['osm_id'] = osm_id
    if lng:
        querystring['lng'] = lng
    if lat:
        querystring['lat'] = lat
    if geo:
        querystring['geo'] = geo
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "tranquilcitydata.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def london_grid_tranquil(lng: str, lat: str, geo: str=None, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "All"
    
    """
    url = f"https://tranquilcitydata.p.rapidapi.com/v1/london/grid/tranquil/"
    querystring = {'lng': lng, 'lat': lat, }
    if geo:
        querystring['geo'] = geo
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "tranquilcitydata.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def london_grid_greeninfrastructure(lng: str, lat: str, geo: str=None, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "All"
    
    """
    url = f"https://tranquilcitydata.p.rapidapi.com/v1/london/grid/greeninfrastructure/"
    querystring = {'lng': lng, 'lat': lat, }
    if geo:
        querystring['geo'] = geo
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "tranquilcitydata.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def london_grid_environmental(lat: str, lng: str, geo: str=None, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "All"
    
    """
    url = f"https://tranquilcitydata.p.rapidapi.com/v1/london/grid/environmental/"
    querystring = {'lat': lat, 'lng': lng, }
    if geo:
        querystring['geo'] = geo
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "tranquilcitydata.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

