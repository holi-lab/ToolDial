import requests
import json
from datetime import date, datetime, timedelta
import os

from typing import Optional, Dict, Union, List


def events_search_paging(lat: str, lon: str, d: str, radius: int=50, p: str=None, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "get events in Italy depending on geoposition and date"
    lat: latitude of geoposition
        lon: longitude of the geoposition
        d: date
        radius: radius in km
        p: page number
        
    """
    url = f"https://zaza81-events-v1.p.rapidapi.com/events_search_paging"
    querystring = {'lat': lat, 'lon': lon, 'd': d, }
    if radius:
        querystring['radius'] = radius
    if p:
        querystring['p'] = p
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "zaza81-events-v1.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

