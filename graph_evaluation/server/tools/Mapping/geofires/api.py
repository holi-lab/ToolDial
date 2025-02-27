import requests
import json
from datetime import date, datetime, timedelta
import os

from typing import Optional, Dict, Union, List


def aggregate(date: str, format: str='geojson', toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Aggregates the broadcasted news related to wildfires using a spatial grid and returns the features as hexagonal bins.
		The date must be in ISO 8601 format, e.g. '2015-03-01'.
		The underlying knowledge graph collects data from '2015-03-01' up to today.
		The format can be geojson or esri."
    
    """
    url = f"https://geofires.p.rapidapi.com/aggregate"
    querystring = {'date': date, }
    if format:
        querystring['format'] = format
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "geofires.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def query(date: str, format: str='geojson', toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Returns the locations related to wildfires.
		The date must be in ISO 8601 format, e.g. '2015-03-01'.
		The underlying knowledge graph collects data from '2015-03-01' up to today.
		The format can be geojson or esri."
    
    """
    url = f"https://geofires.p.rapidapi.com/query"
    querystring = {'date': date, }
    if format:
        querystring['format'] = format
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "geofires.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def articles(date: str='2021-12-31', toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Returns a list of broadcasted articles related to wild fires.
		The date must be in ISO 8601 format, e.g. '2015-03-01'.
		The underlying knowledge graph collects data from '2015-03-01' up to today."
    
    """
    url = f"https://geofires.p.rapidapi.com/articles"
    querystring = {}
    if date:
        querystring['date'] = date
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "geofires.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

