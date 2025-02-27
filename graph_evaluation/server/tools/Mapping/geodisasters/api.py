import requests
import json
from datetime import date, datetime, timedelta
import os

from typing import Optional, Dict, Union, List


def query(is_from: str, to: str, format: str='geojson', toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Returns the most common locations related to natural disasters using a specific date range. The maximum date range is between 2023-05-24 and yesterday."
    
    """
    url = f"https://geodisasters.p.rapidapi.com/query"
    querystring = {'from': is_from, 'to': to, }
    if format:
        querystring['format'] = format
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "geodisasters.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def hotspots(date: str='2023-05-24', format: str='geojson', toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Returns the hotspot locations related to natural disasters.
		
		The date is optional. When not specified, we return the features of the last 24 hours.
		The underlying knowledge graph collects locations since 2023-05-24 and yesterday should be the latest available date.
		The format can be geojson or esri."
    
    """
    url = f"https://geodisasters.p.rapidapi.com/hotspots"
    querystring = {}
    if date:
        querystring['date'] = date
    if format:
        querystring['format'] = format
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "geodisasters.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def aggregate(format: str='geojson', date: str='2023-05-24', toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Aggregates the broadcasted news related to natural disasters using a spatial grid and returns the features as hexagonal bins.
		
		The date is optional. When not specified, we return the features of the last 24 hours.
		The underlying knowledge graph contains locations since 2023-05-24 and yesterday should be the latest available date.
		The format can be geojson or esri."
    
    """
    url = f"https://geodisasters.p.rapidapi.com/aggregate"
    querystring = {}
    if format:
        querystring['format'] = format
    if date:
        querystring['date'] = date
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "geodisasters.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

