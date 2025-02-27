import requests
import json
from datetime import date, datetime, timedelta
import os

from typing import Optional, Dict, Union, List


def cities_in_the_country(code: str, per_page: int=20, format: str=None, page: int=2, population: int=1501, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Returns the list of cities for the specified population. Maximum 1000 cities per page and the default population is greater than 15000"
    code: Country Code (example US, GB, NO)
        per_page: results per page (max 20)
        format: format of the response
        page: page number
        population: The minimal population of the city
        
    """
    url = f"https://countries-cities.p.rapidapi.com/location/country/{code}/city/list"
    querystring = {}
    if per_page:
        querystring['per_page'] = per_page
    if format:
        querystring['format'] = format
    if page:
        querystring['page'] = page
    if population:
        querystring['population'] = population
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "countries-cities.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def country_geojson(code: str, format: str=None, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Returns geoJSON of the country"
    code: Country Code (example US, GB, NO)
        
    """
    url = f"https://countries-cities.p.rapidapi.com/location/country/{code}/geojson"
    querystring = {}
    if format:
        querystring['format'] = format
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "countries-cities.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def countries_list(format: str=None, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Returns the list of all countries with country codes"
    format: format of the response
        
    """
    url = f"https://countries-cities.p.rapidapi.com/location/country/list"
    querystring = {}
    if format:
        querystring['format'] = format
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "countries-cities.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def city_details(geonameid: int, format: str=None, language: str=None, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Returns a detailed information about the city and it's location"
    
    """
    url = f"https://countries-cities.p.rapidapi.com/location/city/{geonameid}"
    querystring = {}
    if format:
        querystring['format'] = format
    if language:
        querystring['language'] = language
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "countries-cities.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def country_details(code: str, format: str=None, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Returns capital, location, timezone, population, currency, languages, phone code of the country"
    code: Country code (example: US, GB, NO)
        format: format of the response
        
    """
    url = f"https://countries-cities.p.rapidapi.com/location/country/{code}"
    querystring = {}
    if format:
        querystring['format'] = format
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "countries-cities.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def cities_nearby(max_population: int=10000, longitude: int=37, format: str=None, per_page: int=10, latitude: int=55, min_population: int=99, radius: int=25, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "list of the cities next to the location"
    
    """
    url = f"https://countries-cities.p.rapidapi.com/location/city/nearby"
    querystring = {}
    if max_population:
        querystring['max_population'] = max_population
    if longitude:
        querystring['longitude'] = longitude
    if format:
        querystring['format'] = format
    if per_page:
        querystring['per_page'] = per_page
    if latitude:
        querystring['latitude'] = latitude
    if min_population:
        querystring['min_population'] = min_population
    if radius:
        querystring['radius'] = radius
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "countries-cities.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

