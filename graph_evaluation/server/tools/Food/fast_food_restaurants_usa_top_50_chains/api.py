import requests
import json
from datetime import date, datetime, timedelta
import os

from typing import Optional, Dict, Union, List


def get_all_city_names(toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "GET all City names"
    
    """
    url = f"https://fast-food-restaurants-usa-top-50-chains.p.rapidapi.com/restaurants-top/all-city"
    querystring = {}
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "fast-food-restaurants-usa-top-50-chains.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def get_all_state_names(toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "GET all {State} names"
    
    """
    url = f"https://fast-food-restaurants-usa-top-50-chains.p.rapidapi.com/restaurants-top/all-state"
    querystring = {}
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "fast-food-restaurants-usa-top-50-chains.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def get_all_chain_names(toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Get all {Chain} names"
    
    """
    url = f"https://fast-food-restaurants-usa-top-50-chains.p.rapidapi.com/restaurants-top/get-all-restaurant-names"
    querystring = {}
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "fast-food-restaurants-usa-top-50-chains.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def get_all_restaurants_by_chain_city_state(page: int, city: str, state: str, restaurantchainname: str, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "**10 results per request. Use page number for Pagination.**"
    
    """
    url = f"https://fast-food-restaurants-usa-top-50-chains.p.rapidapi.com/restaurants-top/{restaurantchainname}/location/state/{state}/city/{city}/{page}"
    querystring = {}
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "fast-food-restaurants-usa-top-50-chains.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def get_all_restaurants_by_chain(restaurantchainname: str, page: int, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "**10 results per request. Use page number for Pagination.**"
    
    """
    url = f"https://fast-food-restaurants-usa-top-50-chains.p.rapidapi.com/restaurants-top/{restaurantchainname}/location/{page}"
    querystring = {}
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "fast-food-restaurants-usa-top-50-chains.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def get_all_restaurants_by_chain_state(state: str, restaurantchainname: str, page: int, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "**10 results per request. Use page number for Pagination.**"
    
    """
    url = f"https://fast-food-restaurants-usa-top-50-chains.p.rapidapi.com/restaurants-top/{restaurantchainname}/location/state/{state}/{page}"
    querystring = {}
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "fast-food-restaurants-usa-top-50-chains.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def get_restaurant_logo_by_chain_name(restaurantchainname: str, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Get Restaurant {Logo} by {Chain} Name"
    
    """
    url = f"https://fast-food-restaurants-usa-top-50-chains.p.rapidapi.com/restaurants-top/{restaurantchainname}/media"
    querystring = {}
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "fast-food-restaurants-usa-top-50-chains.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

