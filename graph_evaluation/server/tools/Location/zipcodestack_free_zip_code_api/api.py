import requests
import json
from datetime import date, datetime, timedelta
import os

from typing import Optional, Dict, Union, List


def getv1status(apikey: str='{YOUR_API_KEY}', toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Returns the remaining requests for the given API key."
    
    """
    url = f"https://zipcodestack-free-zip-code-api.p.rapidapi.com/v1/status"
    querystring = {}
    if apikey:
        querystring['apikey'] = apikey
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "zipcodestack-free-zip-code-api.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def getv1search(codes: str, apikey: str='{YOUR_API_KEY}', country: str='us', toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Search zip codes with this endpoint"
    codes: string Comma-separated list of zip codes.
        country: Two letter country code.
        
    """
    url = f"https://zipcodestack-free-zip-code-api.p.rapidapi.com/v1/search"
    querystring = {'codes': codes, }
    if apikey:
        querystring['apikey'] = apikey
    if country:
        querystring['country'] = country
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "zipcodestack-free-zip-code-api.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def getv1distance(code: str, compare: str, country: str, apikey: str='{YOUR_API_KEY}', unit: str='km', toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Calculates the distance between one and other postalcodes"
    code: string Zip code.
        compare: string The list of zip codes to compare to.
        country: string Two letter country code.
        unit: The unit of distance, can be kilometers or miles. Defaults to km.
        
    """
    url = f"https://zipcodestack-free-zip-code-api.p.rapidapi.com/v1/distance"
    querystring = {'code': code, 'compare': compare, 'country': country, }
    if apikey:
        querystring['apikey'] = apikey
    if unit:
        querystring['unit'] = unit
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "zipcodestack-free-zip-code-api.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

