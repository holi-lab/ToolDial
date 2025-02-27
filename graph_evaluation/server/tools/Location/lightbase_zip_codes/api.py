import requests
import json
from datetime import date, datetime, timedelta
import os

from typing import Optional, Dict, Union, List


def city_zip_codes(zipcode1: str, zipcode2: str, city: str, state: str, key: str, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Get all zip codes within a city."
    
    """
    url = f"https://lightbase-zip-codes.p.rapidapi.com/zip/cityzips/{city}/{state}"
    querystring = {'zipcode1': zipcode1, 'zipcode2': zipcode2, 'key': key, }
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "lightbase-zip-codes.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def distance_calculator(zipcode2: str, zipcode1: str, key: str, zipcode_one: str, zipcode_two: str, unit: str='M', toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Calculate distance between two zip codes."
    
    """
    url = f"https://lightbase-zip-codes.p.rapidapi.com/zip/distance/{zipcode_one}/{zipcode_two}"
    querystring = {'zipcode2': zipcode2, 'zipcode1': zipcode1, 'key': key, }
    if unit:
        querystring['unit'] = unit
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "lightbase-zip-codes.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def find_nearest_cities(radius: str, zipcode: str, key: str, unit: str='M', toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Find cities within a given radius."
    
    """
    url = f"https://lightbase-zip-codes.p.rapidapi.com/zip/nearest/{zipcode}/{radius}"
    querystring = {'key': key, }
    if unit:
        querystring['unit'] = unit
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "lightbase-zip-codes.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def zip_code_info(zipcode: str, key: str, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Get city, state, county, region, and geo information for a zip code."
    
    """
    url = f"https://lightbase-zip-codes.p.rapidapi.com/zip/zipinfo/{zipcode}"
    querystring = {'key': key, }
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "lightbase-zip-codes.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

