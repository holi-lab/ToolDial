import requests
import json
from datetime import date, datetime, timedelta
import os

from typing import Optional, Dict, Union, List


def cities_in_the_country(code: str, format: str=None, per_page: int=10, language: str=None, max_population: int=10000, page: int=1, min_population: int=100, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Returns the list of the cities in the country"
    per_page: Maximum number of results per page is 100
        
    """
    url = f"https://world-geo-data.p.rapidapi.com/countries/{code}/cities"
    querystring = {}
    if format:
        querystring['format'] = format
    if per_page:
        querystring['per_page'] = per_page
    if language:
        querystring['language'] = language
    if max_population:
        querystring['max_population'] = max_population
    if page:
        querystring['page'] = page
    if min_population:
        querystring['min_population'] = min_population
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "world-geo-data.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def cities_in_the_administrative_division(geonameid: int, language: str=None, page: int=1, max_population: int=10000, per_page: int=10, min_population: int=100, format: str=None, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Returns the list of cities in the administrative division"
    per_page: Maximum number of results per page is 100
        
    """
    url = f"https://world-geo-data.p.rapidapi.com/adm-divisions/{geonameid}/cities"
    querystring = {}
    if language:
        querystring['language'] = language
    if page:
        querystring['page'] = page
    if max_population:
        querystring['max_population'] = max_population
    if per_page:
        querystring['per_page'] = per_page
    if min_population:
        querystring['min_population'] = min_population
    if format:
        querystring['format'] = format
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "world-geo-data.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def cities_near_location(per_page: int=9, language: str=None, format: str=None, max_population: int=10000, radius: int=25, page: int=1, latitude: int=55, longitude: int=37, min_population: int=100, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Returns the cities in the specified radius from the geo location"
    per_page: Maximum number of results per page is 100
        
    """
    url = f"https://world-geo-data.p.rapidapi.com/cities/nearby"
    querystring = {}
    if per_page:
        querystring['per_page'] = per_page
    if language:
        querystring['language'] = language
    if format:
        querystring['format'] = format
    if max_population:
        querystring['max_population'] = max_population
    if radius:
        querystring['radius'] = radius
    if page:
        querystring['page'] = page
    if latitude:
        querystring['latitude'] = latitude
    if longitude:
        querystring['longitude'] = longitude
    if min_population:
        querystring['min_population'] = min_population
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "world-geo-data.p.rapidapi.com"
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
    url = f"https://world-geo-data.p.rapidapi.com/cities/{geonameid}"
    querystring = {}
    if format:
        querystring['format'] = format
    if language:
        querystring['language'] = language
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "world-geo-data.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def country_details(code: str, format: str=None, language: str='en,ru,zh,es,ar,fr,fa,ja,pl,it,pt,de', toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Information about the country: area_size, country, continent, timezone, currency, population, flag, dependent territories, wikipedia reference totals of the cities and the first-level divisions"
    
    """
    url = f"https://world-geo-data.p.rapidapi.com/countries/{code}"
    querystring = {}
    if format:
        querystring['format'] = format
    if language:
        querystring['language'] = language
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "world-geo-data.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def country_administrative_divisions(code: str, format: str=None, per_page: str=None, language: str='en,ru,zh,es,ar,fr,fa,ja,pl,it,pt,de', page: int=1, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Returns the list of administrative divisions of the country"
    per_page: Maximum number of results per page is 100
        
    """
    url = f"https://world-geo-data.p.rapidapi.com/countries/{code}/adm-divisions"
    querystring = {}
    if format:
        querystring['format'] = format
    if per_page:
        querystring['per_page'] = per_page
    if language:
        querystring['language'] = language
    if page:
        querystring['page'] = page
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "world-geo-data.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def country_iso_divisions(code: str, format: str=None, language: str=None, per_page: int=10, page: int=1, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Returns the list of divisions based on the ISO Standard"
    per_page: Maximum number of results per page is 100
        
    """
    url = f"https://world-geo-data.p.rapidapi.com/countries/{code}/iso-divisions"
    querystring = {}
    if format:
        querystring['format'] = format
    if language:
        querystring['language'] = language
    if per_page:
        querystring['per_page'] = per_page
    if page:
        querystring['page'] = page
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "world-geo-data.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def country_iso_sub_divisions(iso_code: str, format: str=None, page: int=1, language: str=None, per_page: int=10, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Returns the list of sub divisions in the specific division based on the ISO Standard"
    per_page: Maximum number of results per page is 100
        
    """
    url = f"https://world-geo-data.p.rapidapi.com/countries/{code}/iso-divisions/{iso_code}"
    querystring = {}
    if format:
        querystring['format'] = format
    if page:
        querystring['page'] = page
    if language:
        querystring['language'] = language
    if per_page:
        querystring['per_page'] = per_page
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "world-geo-data.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def country_administrative_division_details(geonameid: int, format: str=None, language: str=None, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Returns the detailed information about administrative division"
    
    """
    url = f"https://world-geo-data.p.rapidapi.com/adm-divisions/{geonameid}"
    querystring = {}
    if format:
        querystring['format'] = format
    if language:
        querystring['language'] = language
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "world-geo-data.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def country_iso_division_details(iso_code: str, format: str=None, languages: str=None, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Returns the detailed information about  ISO division"
    
    """
    url = f"https://world-geo-data.p.rapidapi.com/iso-divisions/{iso_code}"
    querystring = {}
    if format:
        querystring['format'] = format
    if languages:
        querystring['languages'] = languages
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "world-geo-data.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def country_administrative_sub_divisions(code: str, geonameid: str, per_page: int=10, page: int=1, language: str=None, format: str=None, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Returns the list of administrative sub divisions  of the country based on the geonameid"
    
    """
    url = f"https://world-geo-data.p.rapidapi.com/countries/{code}/adm-divisions/{geonameid}"
    querystring = {}
    if per_page:
        querystring['per_page'] = per_page
    if page:
        querystring['page'] = page
    if language:
        querystring['language'] = language
    if format:
        querystring['format'] = format
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "world-geo-data.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def countries_and_dependent_territories(territories: bool=None, language: str=None, format: str=None, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Returns a list of countries and territories if requested"
    
    """
    url = f"https://world-geo-data.p.rapidapi.com/countries"
    querystring = {}
    if territories:
        querystring['territories'] = territories
    if language:
        querystring['language'] = language
    if format:
        querystring['format'] = format
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "world-geo-data.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def country_geojson(code: str, format: str=None, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Returns the multi-polygon of the country"
    
    """
    url = f"https://world-geo-data.p.rapidapi.com/countries/{code}/geojson"
    querystring = {}
    if format:
        querystring['format'] = format
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "world-geo-data.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

