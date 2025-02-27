import requests
import json
from datetime import date, datetime, timedelta
import os

from typing import Optional, Dict, Union, List


def get_all_countries_paginated(page_size: int, page_number: int, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Get a paginated list of countries."
    
    """
    url = f"https://world-cities6.p.rapidapi.com/countries"
    querystring = {'page_size': page_size, 'page_number': page_number, }
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "world-cities6.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def get_countries_by_region_code(page_size: int, page_number: int, regioncode: str, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Get countries from a region. For example, get a paginated result for all countries in Europe."
    
    """
    url = f"https://world-cities6.p.rapidapi.com/countries/region/{regioncode}"
    querystring = {'page_size': page_size, 'page_number': page_number, }
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "world-cities6.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def get_countries_by_code(countrycode: str, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Get one country by its ISO2 code"
    
    """
    url = f"https://world-cities6.p.rapidapi.com/countries/code/{countrycode}"
    querystring = {}
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "world-cities6.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def get_countries_by_name(page_number: int, page_size: int, countryname: str, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Get a country by its name. This is an autocomplete. This endpoint is ideal for search."
    
    """
    url = f"https://world-cities6.p.rapidapi.com/countries/name/{countryname}"
    querystring = {'page_number': page_number, 'page_size': page_size, }
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "world-cities6.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def get_cities_by_coordinates(page_size: int, page_number: int, lat: int, lng: int, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Returns closest cities from coordinates entered."
    
    """
    url = f"https://world-cities6.p.rapidapi.com/cities/coordinates"
    querystring = {'page_size': page_size, 'page_number': page_number, 'lat': lat, 'lng': lng, }
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "world-cities6.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def get_cities_by_name_autocomplete(page_size: int, page_number: int, cityname: str, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Returns cities from city name.
		Cities that include the city name you are searching for.  This allows for autocomplete features. 
		Results are paginated."
    
    """
    url = f"https://world-cities6.p.rapidapi.com/cities/name/{cityname}"
    querystring = {'page_size': page_size, 'page_number': page_number, }
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "world-cities6.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def get_cities_by_id(cityid: str, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Returns city from city id."
    
    """
    url = f"https://world-cities6.p.rapidapi.com/cities/id/{cityid}"
    querystring = {}
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "world-cities6.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def get_city_by_country_code(page_number: int, page_size: int, countrycode: str, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Returns paginated cities from a country code.
		The country code must be an ISO2 format.
		
		**Example: FR**"
    page_number: Minimum: 1
        page_size: Minimum: 1
Maximum: 100
        
    """
    url = f"https://world-cities6.p.rapidapi.com/cities/name/{countrycode}"
    querystring = {'page_number': page_number, 'page_size': page_size, }
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "world-cities6.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

