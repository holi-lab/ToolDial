import requests
import json
from datetime import date, datetime, timedelta
import os

from typing import Optional, Dict, Union, List


def live_base64_jpg_image(is_id: str, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Returns live, base64 encoded, jpg from the camera."
    id: Camera id to get the live image for
        
    """
    url = f"https://marcoherbst-evercam-public-cameras.p.rapidapi.com/cameras/{is_id}/live.json"
    querystring = {}
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "marcoherbst-evercam-public-cameras.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def live_jpg_image(is_id: str, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Returns live jpg from the camera."
    id: Camera id to get the live image for
        
    """
    url = f"https://marcoherbst-evercam-public-cameras.p.rapidapi.com/cameras/{is_id}/snapshot.jpg"
    querystring = {}
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "marcoherbst-evercam-public-cameras.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def camera_information(is_id: str, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Returns all data for a given camera."
    id: Camera id to get data for
        
    """
    url = f"https://marcoherbst-evercam-public-cameras.p.rapidapi.com/cameras/{is_id}.json"
    querystring = {}
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "marcoherbst-evercam-public-cameras.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def public_cameras(id_contains: str=None, id_ends_with: str=None, id_starts_with: str=None, is_near_to: str=None, within_distance: int=None, case_sensitive: bool=None, limit: int=100, offset: int=0, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Fetch a list of publicly discoverable cameras from within the Evercam system."
    id_contains: Search for cameras whose id contain the given value
        id_ends_with: Search for cameras whose id ends with the given value
        id_starts_with: Search for cameras whose id starts with the given value
        is_near_to: Search for cameras within 1000 meters of a given address or longitude latitiude point
        within_distance: Search for cameras within a greater range of the specified is_near_to point in meters
        case_sensitive: Whether search terms are case sensitive
        limit: Maximum number of results. Defaults to 100, can't be more than 1000
        offset: Offset into the search results Defaults to 0
        
    """
    url = f"https://marcoherbst-evercam-public-cameras.p.rapidapi.com/public/cameras.json"
    querystring = {}
    if id_contains:
        querystring['id_contains'] = id_contains
    if id_ends_with:
        querystring['id_ends_with'] = id_ends_with
    if id_starts_with:
        querystring['id_starts_with'] = id_starts_with
    if is_near_to:
        querystring['is_near_to'] = is_near_to
    if within_distance:
        querystring['within_distance'] = within_distance
    if case_sensitive:
        querystring['case_sensitive'] = case_sensitive
    if limit:
        querystring['limit'] = limit
    if offset:
        querystring['offset'] = offset
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "marcoherbst-evercam-public-cameras.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

