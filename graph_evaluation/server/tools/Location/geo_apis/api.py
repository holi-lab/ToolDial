import requests
import json
from datetime import date, datetime, timedelta
import os

from typing import Optional, Dict, Union, List


def autosuggestion_city_names(page: str='0', size: str='20', name: str='london', countrycode: str='GB', toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "search for cities around the world"
    countrycode: DE,GB,US etc.
        
    """
    url = f"https://geo-apis.p.rapidapi.com/geo"
    querystring = {}
    if page:
        querystring['page'] = page
    if size:
        querystring['size'] = size
    if name:
        querystring['name'] = name
    if countrycode:
        querystring['countryCode'] = countrycode
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "geo-apis.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def zip_code_search(postalcodes: str='14197,10559', size: int=10, countrycodes: str='US,DE', page: int=0, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "find countries, geo location, zip/postal -codes around the world"
    
    """
    url = f"https://geo-apis.p.rapidapi.com/geo"
    querystring = {}
    if postalcodes:
        querystring['postalCodes'] = postalcodes
    if size:
        querystring['size'] = size
    if countrycodes:
        querystring['countryCodes'] = countrycodes
    if page:
        querystring['page'] = page
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "geo-apis.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

