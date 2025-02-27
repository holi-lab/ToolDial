import requests
import json
from datetime import date, datetime, timedelta
import os

from typing import Optional, Dict, Union, List


def get_city_coordinates(name: str, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "This endpoint retrieve only the name and coordinates for the given city name."
    
    """
    url = f"https://cities-of-italy-api.p.rapidapi.com/cities/coordinates/{name}"
    querystring = {}
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "cities-of-italy-api.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def get_city_by_postal_code(cap: str, page: int=1, pagesize: str='10', toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Get all cities that contain the postal code passed as a parameter"
    
    """
    url = f"https://cities-of-italy-api.p.rapidapi.com/cities/byCap/{cap}"
    querystring = {}
    if page:
        querystring['page'] = page
    if pagesize:
        querystring['pageSize'] = pagesize
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "cities-of-italy-api.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def get_cities_by_phone_prefix(phoneprefix: str, pagesize: int=30, page: int=1, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Get all cities that contain the area code passed as a parameter"
    
    """
    url = f"https://cities-of-italy-api.p.rapidapi.com/cities/byPhonePrefix/{phoneprefix}"
    querystring = {}
    if pagesize:
        querystring['pageSize'] = pagesize
    if page:
        querystring['page'] = page
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "cities-of-italy-api.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def get_city_by_name_if_name_contains(name: str, pagesize: int=25, page: int=1, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Get all cities containing the parameter passed in the name."
    
    """
    url = f"https://cities-of-italy-api.p.rapidapi.com/cities/byName/{name}"
    querystring = {}
    if pagesize:
        querystring['pageSize'] = pagesize
    if page:
        querystring['page'] = page
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "cities-of-italy-api.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def get_city_by_exact_name(name: str, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Endpoint that retrieve the single city if matches with the provided name"
    
    """
    url = f"https://cities-of-italy-api.p.rapidapi.com/cities/byExactName/{name}"
    querystring = {}
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "cities-of-italy-api.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def get_all_italian_cities(pagesize: int=25, page: int=1, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Get all italian cities with pagination."
    
    """
    url = f"https://cities-of-italy-api.p.rapidapi.com/cities"
    querystring = {}
    if pagesize:
        querystring['pageSize'] = pagesize
    if page:
        querystring['page'] = page
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "cities-of-italy-api.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def get_city_by_province(province: str, page: int=1, pagesize: str='10', toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "GET all cities belonging to the passed province as a parameter"
    
    """
    url = f"https://cities-of-italy-api.p.rapidapi.com/cities/byProvince/{province}"
    querystring = {}
    if page:
        querystring['page'] = page
    if pagesize:
        querystring['pageSize'] = pagesize
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "cities-of-italy-api.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

