import requests
import json
from datetime import date, datetime, timedelta
import os

from typing import Optional, Dict, Union, List


def country_details_level_1(param: str, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Specified one valid param and retrieve access level 1 information about a country:
		- english name
		- population (in millions)
		
		
		Valid params are country english name, ISO code 2 or ISO code 3"
    param: For the param you must use only:
1. the country english name
2. country ISO code 2
3. country ISO code 3

- The param is not case sensitive.
- Accept spaces between a country name

        
    """
    url = f"https://countries-population3.p.rapidapi.com/countries/{param}/lv1/"
    querystring = {}
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "countries-population3.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def total_population(toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Returns the estimated total population
		
		- Number in millions."
    
    """
    url = f"https://countries-population3.p.rapidapi.com/countries/total_population/"
    querystring = {}
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "countries-population3.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def country_details_level_2(param: str, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Specified one valid param and retrieve access level 2 information about a country:
		- english name
		- ISO code 2
		- numeric code
		- population (in millions)
		
		Valid params are country english name, ISO code 2 or ISO code 3"
    param: For the param you must use only:
1. the country english name
2. country ISO code 2
3. country ISO code 3

- The param is not case sensitive.
- Accept spaces between a country name

        
    """
    url = f"https://countries-population3.p.rapidapi.com/countries/{param}/lv2/"
    querystring = {}
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "countries-population3.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def country_details_level_3(param: str, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Specified one valid param and retrieve access level 3 information about a country:
		
		- english name
		- french name
		- ISO code 2
		- ISO code 3
		- numeric code
		- population (in millions)
		
		Valid params are country english name, ISO code 2 or ISO code 3"
    param: For the param you must use only:
1. the country english name
2. country ISO code 2
3. country ISO code 3

- The param is not case sensitive.
- Accept spaces between a country name

        
    """
    url = f"https://countries-population3.p.rapidapi.com/countries/{param}/lv3/"
    querystring = {}
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "countries-population3.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def all_countries_names(toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Returns all countries english full names. Useful when need to know wich country names to search for."
    
    """
    url = f"https://countries-population3.p.rapidapi.com/countries/names/"
    querystring = {}
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "countries-population3.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

