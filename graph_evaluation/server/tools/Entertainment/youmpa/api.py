import requests
import json
from datetime import date, datetime, timedelta
import os

from typing import Optional, Dict, Union, List


def events(date: str='2013-05-31', coords: str='21.4564545,9.4556456', address: str='Piazza del Duomo, Milano', distance: str='5', start: str='0', limit: str='25', toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    ""
    date: If you are looking for events occurring in a specific date
        coords: Latitude and Longitude (remember one of "coord" and "address" must be passed)
        address: The address (even just the city name) to find events around (remember one of "coord" and "address" must be passed)
        distance: (in Kilometers) the max distance from "coordinates" or "address"
        start: Pagination start
        limit: Pagination limit
        
    """
    url = f"https://youmpaevents.p.rapidapi.com/event_list"
    querystring = {}
    if date:
        querystring['date'] = date
    if coords:
        querystring['coords'] = coords
    if address:
        querystring['address'] = address
    if distance:
        querystring['distance'] = distance
    if start:
        querystring['start'] = start
    if limit:
        querystring['limit'] = limit
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "youmpaevents.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

