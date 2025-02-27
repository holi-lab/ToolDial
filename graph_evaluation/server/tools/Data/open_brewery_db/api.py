import requests
import json
from datetime import date, datetime, timedelta
import os

from typing import Optional, Dict, Union, List


def get_a_brewery(is_id: str, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Get Brewery by ID"
    id: ID of Brewery
        
    """
    url = f"https://brianiswu-open-brewery-db-v1.p.rapidapi.com/breweries/{is_id}"
    querystring = {}
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "brianiswu-open-brewery-db-v1.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def breweries(by_state: str='NY', by_name: str='cooper', by_type: str='micro', sort: str=None, by_tag: str='patio', toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "List of Breweries"
    by_state: Search by State
        by_name: Search by Name
        by_type: Must be one of: micro, regional, brewpub, large, planning, bar, contract, proprietor
        sort: + for acending order (default order); - for decending order comma-separated. Example: sort=-name,+type,city
        by_tag: Must be one of: dog-friendly, patio, food-service, food-truck, tours
        
    """
    url = f"https://brianiswu-open-brewery-db-v1.p.rapidapi.com/breweries"
    querystring = {}
    if by_state:
        querystring['by_state'] = by_state
    if by_name:
        querystring['by_name'] = by_name
    if by_type:
        querystring['by_type'] = by_type
    if sort:
        querystring['sort'] = sort
    if by_tag:
        querystring['by_tag'] = by_tag
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "brianiswu-open-brewery-db-v1.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def autocomplete(query: str, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    ""
    query: Autocomplete a query
        
    """
    url = f"https://brianiswu-open-brewery-db-v1.p.rapidapi.com/breweries/autocomplete"
    querystring = {'query': query, }
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "brianiswu-open-brewery-db-v1.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def search(query: str='dog', toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Search for a brewery"
    query: Search for a brewery
        
    """
    url = f"https://brianiswu-open-brewery-db-v1.p.rapidapi.com/breweries/search"
    querystring = {}
    if query:
        querystring['query'] = query
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "brianiswu-open-brewery-db-v1.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

