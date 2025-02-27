import requests
import json
from datetime import date, datetime, timedelta
import os

from typing import Optional, Dict, Union, List


def get_sun_hours_by_key(sun: str, key: str, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "get Sun-Hours by key"
    
    """
    url = f"https://weatheronline2-sun-hours-forecast-v1.p.rapidapi.com/api/getforecastdata?SUN=1"
    querystring = {'SUN': sun, 'KEY': key, }
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "weatheronline2-sun-hours-forecast-v1.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def list_of_all_countries(toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "List of all Countries"
    
    """
    url = f"https://weatheronline2-sun-hours-forecast-v1.p.rapidapi.com/api/countrycitylist"
    querystring = {}
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "weatheronline2-sun-hours-forecast-v1.p.rapidapi.com"
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
    url = f"https://weatheronline2-sun-hours-forecast-v1.p.rapidapi.com/api/getlocation"
    querystring = {}
    if city:
        querystring['city'] = city
    if zipcode:
        querystring['ZIPCODE'] = zipcode
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "weatheronline2-sun-hours-forecast-v1.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def list_of_cities_in_one_country(country: str, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "List of cities in one country"
    
    """
    url = f"https://weatheronline2-sun-hours-forecast-v1.p.rapidapi.com/api/countrycitylist?COUNTRY={country}"
    querystring = {}
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "weatheronline2-sun-hours-forecast-v1.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

