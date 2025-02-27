import requests
import json
from datetime import date, datetime, timedelta
import os

from typing import Optional, Dict, Union, List


def search_by_coordinates_point(lng: str, lat: str, query: str=None, min_kw: str=None, max_kw: str=None, limit: str='20', available: str=None, type: str=None, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Search for EV charging stations near specific geographic coordinates point."
    lng: Longitude of the geographic coordinates point to search near by.
        lat: Latitude of the geographic coordinates point to search near by.
        query: Return EV charging stations matching a specific query or keyword.

**e.g.** *`evgo`*
**e.g.** *`tesla`*
**e.g.** *`chargepoint`*
        min_kw: Restrict the result to the availability for connectors with a specific minimal value of power in kilowatts (closed interval - with that value).
        max_kw: Restrict the result to the availability for connectors with a specific maximal value of power in kilowatts (closed interval - with that value).
        limit: Maximum number of results to return.

**Allowed values:** *1-500*.
**Default:** *20*.
        available: Find EV charging stations with an available connector.
        type: Return EV charging stations with connectors of one or more types, specified as a comma (,) delimited list of connector types to match.

**e.g.** *`CHAdeMO`*
**e.g.** *`J1772,CCS`*
        
    """
    url = f"https://ev-charge-finder.p.rapidapi.com/search-by-coordinates-point"
    querystring = {'lng': lng, 'lat': lat, }
    if query:
        querystring['query'] = query
    if min_kw:
        querystring['min_kw'] = min_kw
    if max_kw:
        querystring['max_kw'] = max_kw
    if limit:
        querystring['limit'] = limit
    if available:
        querystring['available'] = available
    if type:
        querystring['type'] = type
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "ev-charge-finder.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def search_by_location(near: str, query: str=None, max_kw: str=None, limit: str='20', type: str=None, min_kw: str=None, available: str=None, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Search for EV charging stations near a specific location specified as a free-form location query (e.g. *San Francisco, CA, USA*)."
    near: Free-form location query to search near by (e.g. *San Francisco, CA, USA*).
        query: Return EV charging stations matching a specific query or keyword.

**e.g.** *`evgo`*
**e.g.** *`tesla`*
**e.g.** *`chargepoint`*
        max_kw: Restrict the result to the availability for connectors with a specific maximum value of power in kilowatts (closed interval - with that value).
        limit: Maximum number of results to return.

**Allowed values:** *1-500*.
**Default:** *20*.
        type: Return EV charging stations with connectors of one or more types, specified as a comma (,) delimited list of connector types to match.

**e.g.** *`CHAdeMO`*
**e.g.** *`J1772,CCS`*
        min_kw: Restrict the result to the availability for connectors with a specific minimal value of power in kilowatts (closed interval - with that value).
        available: Only return EV charging stations with an available connector.
        
    """
    url = f"https://ev-charge-finder.p.rapidapi.com/search-by-location"
    querystring = {'near': near, }
    if query:
        querystring['query'] = query
    if max_kw:
        querystring['max_kw'] = max_kw
    if limit:
        querystring['limit'] = limit
    if type:
        querystring['type'] = type
    if min_kw:
        querystring['min_kw'] = min_kw
    if available:
        querystring['available'] = available
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "ev-charge-finder.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

