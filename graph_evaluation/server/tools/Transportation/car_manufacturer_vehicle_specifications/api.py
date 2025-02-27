import requests
import json
from datetime import date, datetime, timedelta
import os

from typing import Optional, Dict, Union, List


def cars_search_any_field(toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "search category title, change category to field name to search search from fields available replace category with field name and search_value=value with the search value. Cars. You can change **make ** to any available field to search.
		
		This example will show all cars with the make Volvo"
    
    """
    url = f"https://car-manufacturer-vehicle-specifications.p.rapidapi.com/cars/search_list/make/?search_value=Volvo"
    querystring = {}
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "car-manufacturer-vehicle-specifications.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def cars_list_all_fields_from_multiple_items(toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Pass in multiple items into this search and return results from Cars. max = number items to return.
		
		This example searches for the make Volvo and the model XC90 showing a max of 10 results."
    
    """
    url = f"https://car-manufacturer-vehicle-specifications.p.rapidapi.com/cars/search_multi/?make=Volvo&model=XC90&max=10"
    querystring = {}
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "car-manufacturer-vehicle-specifications.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def cars_distinct_list_of_any_field_values(toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "for example if you want to get a distinct list of make that exist from Cars. this should show a distinct list in json array.
		
		you can change the value make to any available field to return distinct values"
    
    """
    url = f"https://car-manufacturer-vehicle-specifications.p.rapidapi.com/cars/search_distinct/make"
    querystring = {}
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "car-manufacturer-vehicle-specifications.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def cars_latest_50_100_items(toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "show the latest items from the range starting at 50 and showing 100 items from Cars"
    
    """
    url = f"https://car-manufacturer-vehicle-specifications.p.rapidapi.com/cars/latest/50/100/"
    querystring = {}
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "car-manufacturer-vehicle-specifications.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def cars_latest_50_items(toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "show the latest 50 items from Cars"
    
    """
    url = f"https://car-manufacturer-vehicle-specifications.p.rapidapi.com/cars/latest/start/50/"
    querystring = {}
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "car-manufacturer-vehicle-specifications.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def cars_list_items_fields_available(toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Lists all available fields in cars"
    
    """
    url = f"https://car-manufacturer-vehicle-specifications.p.rapidapi.com/cars/list_fields/"
    querystring = {}
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "car-manufacturer-vehicle-specifications.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

