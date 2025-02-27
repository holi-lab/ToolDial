import requests
import json
from datetime import date, datetime, timedelta
import os

from typing import Optional, Dict, Union, List


def cities_search(q: str, page: int=None, country_code: str=None, sortorder: str=None, sortby: str=None, country_name: str=None, country_id: int=None, limit: int=None, state_code: str=None, state_id: int=None, state_name: str=None, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "This endpoint allows you to search for specific cities by name. It returns a list of cities that match the specified search query."
    
    """
    url = f"https://city-and-state-search-api.p.rapidapi.com/cities/search"
    querystring = {'q': q, }
    if page:
        querystring['page'] = page
    if country_code:
        querystring['country_code'] = country_code
    if sortorder:
        querystring['sortOrder'] = sortorder
    if sortby:
        querystring['sortBy'] = sortby
    if country_name:
        querystring['country_name'] = country_name
    if country_id:
        querystring['country_id'] = country_id
    if limit:
        querystring['limit'] = limit
    if state_code:
        querystring['state_code'] = state_code
    if state_id:
        querystring['state_id'] = state_id
    if state_name:
        querystring['state_name'] = state_name
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "city-and-state-search-api.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def city_details(is_id: str, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "This endpoint allows you to retrieve detailed information about a specific city by ID"
    
    """
    url = f"https://city-and-state-search-api.p.rapidapi.com/cities/{is_id}"
    querystring = {}
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "city-and-state-search-api.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def states_details(is_id: str, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "This endpoint allows you to retrieve detailed information about a specific state by ID"
    
    """
    url = f"https://city-and-state-search-api.p.rapidapi.com/states/{is_id}"
    querystring = {}
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "city-and-state-search-api.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def states_search(q: str, limit: str='50', sortorder: str=None, sortby: str=None, country_id: str=None, country_code: str=None, country_name: str=None, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "This endpoint allows you to search for specific states by name. It returns a list of states that match the specified search query. You can also set country code or country name to search states in that country only."
    
    """
    url = f"https://city-and-state-search-api.p.rapidapi.com/states/search"
    querystring = {'q': q, }
    if limit:
        querystring['limit'] = limit
    if sortorder:
        querystring['sortOrder'] = sortorder
    if sortby:
        querystring['sortBy'] = sortby
    if country_id:
        querystring['country_id'] = country_id
    if country_code:
        querystring['country_code'] = country_code
    if country_name:
        querystring['country_name'] = country_name
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "city-and-state-search-api.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def states_list(country_id: str=None, sortorder: str=None, sortby: str=None, limit: int=None, country_code: str='IN', country_name: str=None, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "This endpoint returns a list of States .It returns basic information about each state, such as name, code, and country. You can also use this endpoint to retrieve a list of states filtered by various criteria such as country."
    
    """
    url = f"https://city-and-state-search-api.p.rapidapi.com/states"
    querystring = {}
    if country_id:
        querystring['country_id'] = country_id
    if sortorder:
        querystring['sortOrder'] = sortorder
    if sortby:
        querystring['sortBy'] = sortby
    if limit:
        querystring['limit'] = limit
    if country_code:
        querystring['country_code'] = country_code
    if country_name:
        querystring['country_name'] = country_name
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "city-and-state-search-api.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def country_details(country: str, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "This endpoint allows you to retrieve detailed information about a specific country by ID or his name."
    country: It can be country name, iso2 or id.
        
    """
    url = f"https://city-and-state-search-api.p.rapidapi.com/countries/{country}"
    querystring = {}
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "city-and-state-search-api.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def countries_search(q: str, sortorder: str=None, sortby: str=None, limit: int=None, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "This endpoint allows you to search for specific countries by name. It returns a list of countries that match the specified search query."
    
    """
    url = f"https://city-and-state-search-api.p.rapidapi.com/countries/search"
    querystring = {'q': q, }
    if sortorder:
        querystring['sortOrder'] = sortorder
    if sortby:
        querystring['sortBy'] = sortby
    if limit:
        querystring['limit'] = limit
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "city-and-state-search-api.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def countries_list(sortorder: str=None, sortby: str=None, limit: int=None, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "This endpoint returns a list of countries."
    
    """
    url = f"https://city-and-state-search-api.p.rapidapi.com/countries"
    querystring = {}
    if sortorder:
        querystring['sortOrder'] = sortorder
    if sortby:
        querystring['sortBy'] = sortby
    if limit:
        querystring['limit'] = limit
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "city-and-state-search-api.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def search(q: str, page: int=None, limit: int=None, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "This endpoint allows you to search for locations (cities, states, or countries) by name. It returns a list of locations that match the specified search query."
    
    """
    url = f"https://city-and-state-search-api.p.rapidapi.com/search"
    querystring = {'q': q, }
    if page:
        querystring['page'] = page
    if limit:
        querystring['limit'] = limit
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "city-and-state-search-api.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

