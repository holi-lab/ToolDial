import requests
import json
from datetime import date, datetime, timedelta
import os

from typing import Optional, Dict, Union, List


def property_search(town_id: str, high_price_range: str=None, property_type: str=None, opt: int=None, page: str='1', low_price_range: str=None, bedrooms: str=None, repo: int=None, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Search for property for-sale listings in a geographical area. Returns the property features, image url,  and coordinates."
    town_id: List of town names can be found on [Puerto Rico Towns Coverage | Area o Pueblos](https://bit.ly/3RweCGr)
        high_price_range: Maximum list price in USD
        property_type: One of the following options: apartamento|apartamento/walkup|casa|comercial|finca|multifamiliar|solar. Default is all.
        opt: Options filter. Set it to 1 if needed.
        page: Page number if at the previous response max_page > 1
        low_price_range: Minimum list price in USD
        bedrooms: One of the following options: 1|2|3|4|5|>6. Default is all.
        repo: Repossessed filter. Set it to 1 if needed.
        
    """
    url = f"https://puerto-rico-real-estate.p.rapidapi.com/property/search"
    querystring = {'town_id': town_id, }
    if high_price_range:
        querystring['high_price_range'] = high_price_range
    if property_type:
        querystring['property_type'] = property_type
    if opt:
        querystring['opt'] = opt
    if page:
        querystring['page'] = page
    if low_price_range:
        querystring['low_price_range'] = low_price_range
    if bedrooms:
        querystring['bedrooms'] = bedrooms
    if repo:
        querystring['repo'] = repo
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "puerto-rico-real-estate.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def short_term_rental_search(town_id: str, bedrooms: str=None, high_price_range: str=None, property_type: str=None, page: str='1', low_price_range: str=None, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Search for short-term rental listings in a geographical area. Returns the property features, image url,  and coordinates."
    town_id: List of town names can be found on [Puerto Rico Towns Coverage | Area o Pueblos](https://bit.ly/3RweCGr)
        bedrooms: One of the following options: 1|2|3|4|5|>6. Default is all.
        high_price_range: Maximum list price in USD
        property_type: One of the following options: apartamento|apartamento/walkup|casa|comercial|finca|multifamiliar|solar. Default is all.
        page: Page number if at the previous response max_page > 1
        low_price_range: Minimum list price in USD
        
    """
    url = f"https://puerto-rico-real-estate.p.rapidapi.com/short_term_rental/search"
    querystring = {'town_id': town_id, }
    if bedrooms:
        querystring['bedrooms'] = bedrooms
    if high_price_range:
        querystring['high_price_range'] = high_price_range
    if property_type:
        querystring['property_type'] = property_type
    if page:
        querystring['page'] = page
    if low_price_range:
        querystring['low_price_range'] = low_price_range
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "puerto-rico-real-estate.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def long_term_rental_search(town_id: str, property_type: str=None, high_price_range: str=None, bedrooms: str=None, low_price_range: str=None, page: str='1', toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Search for long-term rental listings in a geographical area. Returns the property features, image url,  and coordinates."
    town_id: List of town names can be found on [Puerto Rico Towns Coverage | Area o Pueblos](https://bit.ly/3RweCGr)
        property_type: One of the following options: apartamento|apartamento/walkup|casa|comercial|finca|multifamiliar|solar. Default is all.
        high_price_range: Maximum list price in USD
        bedrooms: One of the following options: 1|2|3|4|5|>6. Default is all.
        low_price_range: Minimum list price in USD
        page: Page number if at the previous response max_page > 1
        
    """
    url = f"https://puerto-rico-real-estate.p.rapidapi.com/long_term_rental/search"
    querystring = {'town_id': town_id, }
    if property_type:
        querystring['property_type'] = property_type
    if high_price_range:
        querystring['high_price_range'] = high_price_range
    if bedrooms:
        querystring['bedrooms'] = bedrooms
    if low_price_range:
        querystring['low_price_range'] = low_price_range
    if page:
        querystring['page'] = page
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "puerto-rico-real-estate.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def rental_town_by_id(toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Search all long-term and short-term rental town options"
    
    """
    url = f"https://puerto-rico-real-estate.p.rapidapi.com/long_term_rental/filters/towns"
    querystring = {}
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "puerto-rico-real-estate.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def property_town_by_id(toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Search all town options"
    
    """
    url = f"https://puerto-rico-real-estate.p.rapidapi.com/property/filters/towns"
    querystring = {}
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "puerto-rico-real-estate.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

