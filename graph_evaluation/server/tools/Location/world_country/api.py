import requests
import json
from datetime import date, datetime, timedelta
import os

from typing import Optional, Dict, Union, List


def get_regions(keyword: str='asia', perpage: int=50, is_id: str=None, page: int=1, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Get region list"
    keyword: search Region by keyword
        is_id: id of Region
        
    """
    url = f"https://world-country.p.rapidapi.com/get/regions"
    querystring = {}
    if keyword:
        querystring['keyword'] = keyword
    if perpage:
        querystring['perpage'] = perpage
    if is_id:
        querystring['id'] = is_id
    if page:
        querystring['page'] = page
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "world-country.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def get_sub_regions(keyword: str='asia', perpage: int=50, page: int=1, is_id: str=None, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Get sub region list"
    keyword: search Subregion by keyword
        is_id: id of Subregion
        
    """
    url = f"https://world-country.p.rapidapi.com/get/subregions"
    querystring = {}
    if keyword:
        querystring['keyword'] = keyword
    if perpage:
        querystring['perpage'] = perpage
    if page:
        querystring['page'] = page
    if is_id:
        querystring['id'] = is_id
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "world-country.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def get_time_zones(is_id: str=None, keyword: str='asia', page: int=1, perpage: int=50, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Get Time Zone list"
    is_id: id of Time Zone
        keyword: search Time Zone by keyword
        
    """
    url = f"https://world-country.p.rapidapi.com/get/timezones"
    querystring = {}
    if is_id:
        querystring['id'] = is_id
    if keyword:
        querystring['keyword'] = keyword
    if page:
        querystring['page'] = page
    if perpage:
        querystring['perpage'] = perpage
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "world-country.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def get_cities(keyword: str='asia', is_id: str=None, page: int=1, perpage: int=50, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Get City list"
    keyword: search City by keyword
        is_id: id of City
        
    """
    url = f"https://world-country.p.rapidapi.com/get/cities"
    querystring = {}
    if keyword:
        querystring['keyword'] = keyword
    if is_id:
        querystring['id'] = is_id
    if page:
        querystring['page'] = page
    if perpage:
        querystring['perpage'] = perpage
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "world-country.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def get_states(keyword: str='asia', perpage: int=50, city_id: str=None, is_id: str=None, page: int=1, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Get State list"
    keyword: search State by keyword
        city_id: Find State by city id
        is_id: id of State
        
    """
    url = f"https://world-country.p.rapidapi.com/get/states"
    querystring = {}
    if keyword:
        querystring['keyword'] = keyword
    if perpage:
        querystring['perpage'] = perpage
    if city_id:
        querystring['city_id'] = city_id
    if is_id:
        querystring['id'] = is_id
    if page:
        querystring['page'] = page
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "world-country.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def get_countries(keyword: str='ind', subregion_id: str=None, state_id: str=None, timezone_id: str=None, region_id: str=None, perpage: int=50, is_id: str=None, page: int=1, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Get Country list"
    keyword: search Country by keyword
        subregion_id: Find Countries by subregion id
        state_id: Find Countries by state id
        timezone_id: Find Countries by timezone id
        region_id: Find Countries by region id
        is_id: id of Country
        
    """
    url = f"https://world-country.p.rapidapi.com/get/countries"
    querystring = {}
    if keyword:
        querystring['keyword'] = keyword
    if subregion_id:
        querystring['subregion_id'] = subregion_id
    if state_id:
        querystring['state_id'] = state_id
    if timezone_id:
        querystring['timezone_id'] = timezone_id
    if region_id:
        querystring['region_id'] = region_id
    if perpage:
        querystring['perpage'] = perpage
    if is_id:
        querystring['id'] = is_id
    if page:
        querystring['page'] = page
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "world-country.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

