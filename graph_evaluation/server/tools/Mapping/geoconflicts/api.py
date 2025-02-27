import requests
import json
from datetime import date, datetime, timedelta
import os

from typing import Optional, Dict, Union, List


def cluster(format: str='esri', date: str='2022-02-24', toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Creates spatial clusters using the armed conflict events and returns the features as cluster polygons. You must define a specific date intersecting the valid date extent. The date extent endpoint returns the current date extent. The format can be GeoJSON or Esri JSON."
    format: Represents the supported output formats.
        
    """
    url = f"https://geoconflicts.p.rapidapi.com/cluster"
    querystring = {}
    if format:
        querystring['format'] = format
    if date:
        querystring['date'] = date
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "geoconflicts.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def aggregate(date: str, format: str='esri', toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Aggregates the armed conflict events using a spatial grid and returns the features as hexagonal bins. You must define a specific date intersecting the valid date extent. The date extent endpoint returns the current date extent. The format can be GeoJSON or Esri JSON."
    format: Represents the supported output formats.
        
    """
    url = f"https://geoconflicts.p.rapidapi.com/aggregate"
    querystring = {'date': date, }
    if format:
        querystring['format'] = format
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "geoconflicts.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def dateextent(toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Returns the valid date extent of the armed conflict events as a JSON result."
    
    """
    url = f"https://geoconflicts.p.rapidapi.com/dateExtent"
    querystring = {}
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "geoconflicts.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def query(format: str='esri', date: str='2022-02-24', toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Queries the armed conflict events and returns the events as features. You must define a specific date intersecting the valid date extent. The date extent endpoint returns the current date extent. The format can be GeoJSON or Esri JSON."
    format: Represents the supported output formats.
        
    """
    url = f"https://geoconflicts.p.rapidapi.com/query"
    querystring = {}
    if format:
        querystring['format'] = format
    if date:
        querystring['date'] = date
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "geoconflicts.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def extent(toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Returns the spatial extent of the armed conflict events as a JSON result."
    
    """
    url = f"https://geoconflicts.p.rapidapi.com/extent"
    querystring = {}
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "geoconflicts.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def count(toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Returns the number of armed conflict events as a JSON result."
    
    """
    url = f"https://geoconflicts.p.rapidapi.com/count"
    querystring = {}
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "geoconflicts.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

