import requests
import json
from datetime import date, datetime, timedelta
import os

from typing import Optional, Dict, Union, List


def view_food_item_by_id(toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "This request enables clients to retrieve detailed information about a specific food item based on its unique identifier (ID)
		
		API request sent to [https://indnutrientsapi.tech/food/646e44df0e77ec175b88cf32](https://indnutrientsapi.tech/food/646e44df0e77ec175b88cf32)"
    
    """
    url = f"https://ind-nutrient-api1.p.rapidapi.com/food/646e44df0e77ec175b88cf32"
    querystring = {}
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "ind-nutrient-api1.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def view_all_food_items(limit: int=10, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "The request allows clients to retrieve a comprehensive list of all available food items.
		
		API request sent to [https://indnutrientsapi.tech/food](https://indnutrientsapi.tech/food)"
    limit: limit the length of response
        
    """
    url = f"https://ind-nutrient-api1.p.rapidapi.com/food"
    querystring = {}
    if limit:
        querystring['limit'] = limit
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "ind-nutrient-api1.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def view_food_items_by_core(toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "This request allows clients to retrieve detailed information about a food item by filtering by query param `core`.
		
		Clients can include multiple cores in a single request. For example, by sending a request to https://indnutrientsapi.tech/food?core=chicken,dal,paneer, the API will return food items associated with the specified cores: chicken, dal, and paneer.
		
		API request sent to [https://indnutrientsapi.tech/food?core=chicken](https://indnutrientsapi.tech/food?core=chicken)"
    
    """
    url = f"https://ind-nutrient-api1.p.rapidapi.com/food?core=chicken"
    querystring = {}
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "ind-nutrient-api1.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def view_all_types_with_their_food_items(toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "This request allows clients to retrieve a list of all available types along with the food items associated with each type.
		
		API request sent to [https://indnutrientsapi.tech/food/type](https://indnutrientsapi.tech/food/type)"
    
    """
    url = f"https://ind-nutrient-api1.p.rapidapi.com/food/type"
    querystring = {}
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "ind-nutrient-api1.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def view_all_cores_with_their_food_items(toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "This request allows clients to retrieve a list of all available cores along with the food items associated with each core.
		
		API request sent to [https://indnutrientsapi.tech/food/core](https://indnutrientsapi.tech/food/core)"
    
    """
    url = f"https://ind-nutrient-api1.p.rapidapi.com/food/core"
    querystring = {}
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "ind-nutrient-api1.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def view_food_items_by_type(toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "This request allows clients to retrieve detailed information about a food item by filtering by query param `type`.
		
		API request sent to [https://indnutrientsapi.tech/food?type=non-vegetarian](https://indnutrientsapi.tech/food?type=non-vegetarian)"
    
    """
    url = f"https://ind-nutrient-api1.p.rapidapi.com/food?type=non-vegetarian"
    querystring = {}
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "ind-nutrient-api1.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def view_food_item_by_name(toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "This request allows clients to retrieve detailed information about a food item by specifying by its `name`.
		
		When making the request, ensure the food item's name is written with hyphens ("-") separating the words. For instance, "Paneer Butter Masala" should be written as "paneer-butter-masala" in the request.
		
		API request sent to [https://indnutrientsapi.tech/food/name/paneer-butter-masala](https://indnutrientsapi.tech/food/name/paneer-butter-masala)"
    
    """
    url = f"https://ind-nutrient-api1.p.rapidapi.com/food/name/paneer-butter-masala"
    querystring = {}
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "ind-nutrient-api1.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

