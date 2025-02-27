import requests
import json
from datetime import date, datetime, timedelta
import os

from typing import Optional, Dict, Union, List


def get_city_detail(name: str, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Get details of a city from its name."
    
    """
    url = f"https://geoapi13.p.rapidapi.com/v1/city/{name}"
    querystring = {}
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "geoapi13.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def get_cities_in_a_country(country_iso2: str, page_size: int=20, population_min: int=0, next_token: str=None, population_max: int=100000000, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Get cities in a country based on population."
    
    """
    url = f"https://geoapi13.p.rapidapi.com/v1/country/{country_iso2}/city/list"
    querystring = {}
    if page_size:
        querystring['page_size'] = page_size
    if population_min:
        querystring['population_min'] = population_min
    if next_token:
        querystring['next_token'] = next_token
    if population_max:
        querystring['population_max'] = population_max
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "geoapi13.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def get_country_details(name: str, prefix_match: bool=None, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Get details of country from its name and prefix."
    
    """
    url = f"https://geoapi13.p.rapidapi.com/v1/country/{name}"
    querystring = {}
    if prefix_match:
        querystring['prefix_match'] = prefix_match
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "geoapi13.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

