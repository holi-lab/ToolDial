import requests
import json
from datetime import date, datetime, timedelta
import os

from typing import Optional, Dict, Union, List


def predictions(year: int=None, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Get all predictions for a given year."
    year: A calendar year
        
    """
    url = f"https://groundhog-day-api.p.rapidapi.com/api/v1/predictions"
    querystring = {}
    if year:
        querystring['year'] = year
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "groundhog-day-api.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def groundhogs(isgroundhog: str='1', country: str='Canada or USA', toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Returns all prognosticating animals with their known predictions."
    isgroundhog: Filter groundhogs by type (actual, alive groundhogs, or other prognosticators)
        country: Filter groundhogs by country of origin (USA or Canada).
        
    """
    url = f"https://groundhog-day-api.p.rapidapi.com/api/v1/groundhogs"
    querystring = {}
    if isgroundhog:
        querystring['isGroundhog'] = isgroundhog
    if country:
        querystring['country'] = country
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "groundhog-day-api.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def root(toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Returns a welcome message."
    
    """
    url = f"https://groundhog-day-api.p.rapidapi.com/api/v1"
    querystring = {}
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "groundhog-day-api.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def spec(toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Gets the schema for the JSON API as a yaml file."
    
    """
    url = f"https://groundhog-day-api.p.rapidapi.com/api/v1/spec"
    querystring = {}
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "groundhog-day-api.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def groundhog(slug: str, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Returns a prognosticating animal and its known predictions."
    slug: Groundhog name in kebab-case: (eg, lucy-the-lobster)
        
    """
    url = f"https://groundhog-day-api.p.rapidapi.com/api/v1/groundhogs/{slug}"
    querystring = {}
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "groundhog-day-api.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

