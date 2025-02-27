import requests
import json
from datetime import date, datetime, timedelta
import os

from typing import Optional, Dict, Union, List


def get_uv_index_by_key(key: str, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "get UV Index by key"
    
    """
    url = f"https://weatheronline2-uv-index-v1.p.rapidapi.com/api/getforecastdata"
    querystring = {'KEY': key, }
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "weatheronline2-uv-index-v1.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def list_of_all_available_cities_in_one_country(country: str, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "List of all available cities in one country"
    
    """
    url = f"https://weatheronline2-uv-index-v1.p.rapidapi.com/api/countrycitylist"
    querystring = {'COUNTRY': country, }
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "weatheronline2-uv-index-v1.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def list_of_all_available_countries(toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "List of all available countries"
    
    """
    url = f"https://weatheronline2-uv-index-v1.p.rapidapi.com/api/countrycitylist"
    querystring = {}
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "weatheronline2-uv-index-v1.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def search_location_by_name_or_zip_code(city: str=None, zipcode: str=None, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Search location by Name or zip code and get the key for the forecast"
    city: name of a city
        zipcode: zip code of the city
        
    """
    url = f"https://weatheronline2-uv-index-v1.p.rapidapi.com/api/getlocation"
    querystring = {}
    if city:
        querystring['city'] = city
    if zipcode:
        querystring['zipcode'] = zipcode
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "weatheronline2-uv-index-v1.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

