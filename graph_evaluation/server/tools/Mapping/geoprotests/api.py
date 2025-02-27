import requests
import json
from datetime import date, datetime, timedelta
import os

from typing import Optional, Dict, Union, List


def aggregate(date: str='2021-12-31', format: str='geojson', toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Aggregates the broadcasted news related to protests/demonstrations using a spatial grid and returns the features as hexagonal bins.
		The date is optional. When not specified, we return the features of the last 24 hours.
		The underlying hosted feature service saves the last 90 days and yesterday should be the latest available date.
		The format can be geojson or esri."
    
    """
    url = f"https://geoprotests.p.rapidapi.com/aggregate"
    querystring = {}
    if date:
        querystring['date'] = date
    if format:
        querystring['format'] = format
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "geoprotests.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def hotspots(date: str='2021-12-31', format: str='geojson', toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Returns the hotspot locations related to protests/demonstrations.
		The date is optional. When not specified, we return the features of the last 24 hours.
		The underlying hosted feature service saves the last 90 days and yesterday should be the latest available date.
		The format can be geojson or esri."
    
    """
    url = f"https://geoprotests.p.rapidapi.com/hotspots"
    querystring = {}
    if date:
        querystring['date'] = date
    if format:
        querystring['format'] = format
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "geoprotests.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def articles(date: str='2021-12-31', toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Returns a list of broadcasted articles related to protests/demonstrations.
		The date is optional. When not specified, we return the articles of the last 24 hours.
		The underlying web service saves the last 90 days and yesterday should be the latest available date."
    
    """
    url = f"https://geoprotests.p.rapidapi.com/articles"
    querystring = {}
    if date:
        querystring['date'] = date
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "geoprotests.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

