import requests
import json
from datetime import date, datetime, timedelta
import os

from typing import Optional, Dict, Union, List


def places_by_type(type: str, lat1: str='48.3817', geometryid: str=None, lon1: str='10.8728', limit: str='5', lat2: str='48.3436', lon2: str='10.9301', toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Find places by category and location (bounding box or area). Returns name, location, address and other information."
    type: Check accepted types at https://apidocs.geoapify.com/docs/places/api/api
        lat1: Latitude of top-left bounding box coordinate (lat1,lon1,lat2,lon2 should be provided together)
        geometryid: Alternative filter by free-form geometry (e.g reachability isoline)
        lon1: Longitude of top-left bounding box coordinate (lat1,lon1,lat2,lon2 should be provided together)
        limit: Limit maximum number of results
        lat2: Latitude of bottom-right bounding box coordinate (lat1,lon1,lat2,lon2 should be provided together)
        lon2: Longitude of bottom-right bounding box coordinate (lat1,lon1,lat2,lon2 should be provided together)
        
    """
    url = f"https://places-and-amenities.p.rapidapi.com/v1/places"
    querystring = {'type': type, }
    if lat1:
        querystring['lat1'] = lat1
    if geometryid:
        querystring['geometryId'] = geometryid
    if lon1:
        querystring['lon1'] = lon1
    if limit:
        querystring['limit'] = limit
    if lat2:
        querystring['lat2'] = lat2
    if lon2:
        querystring['lon2'] = lon2
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "places-and-amenities.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

