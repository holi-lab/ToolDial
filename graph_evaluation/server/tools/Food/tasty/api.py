import requests
import json
from datetime import date, datetime, timedelta
import os

from typing import Optional, Dict, Union, List


def recipes_list(size: int, is_from: int, sort: str=None, q: str=None, tags: str='under_30_minutes', toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "List recipes by option filters or name"
    size: Number of items returned per response
        from: The offset of items to be ignored in response for paging
        sort: Leave empty to sort by popular as default OR one of the following : approved_at:desc|approved_at:asc
        q: Name of food or, ingredients to search by
        tags: Get suitable values from /tags/list API
        
    """
    url = f"https://tasty.p.rapidapi.com/recipes/list"
    querystring = {'size': size, 'from': is_from, }
    if sort:
        querystring['sort'] = sort
    if q:
        querystring['q'] = q
    if tags:
        querystring['tags'] = tags
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "tasty.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def tips_list(is_id: int, size: int=30, is_from: int=0, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "This endpoint is used to load tips (reviews)"
    id: The value of recipe id returned in .../recipes/list .../feeds/list .../recipes/list-similarities endpoints
        size: Number of items returned per response
        is_from: The offset of items to be ignored in response for paging
        
    """
    url = f"https://tasty.p.rapidapi.com/tips/list"
    querystring = {'id': is_id, }
    if size:
        querystring['size'] = size
    if is_from:
        querystring['from'] = is_from
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "tasty.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def recipes_detail_deprecated(is_id: int, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Get  more information of recipe if available, such as : ingredients, nutrition info, preparation, etc..."
    id: The id value of any recipe returned in recipes/list API
        
    """
    url = f"https://tasty.p.rapidapi.com/recipes/detail"
    querystring = {'id': is_id, }
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "tasty.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def feeds_list(vegetarian: bool, timezone: str, is_from: int, size: int, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "List latest feeds about new food, recipes,etc..."
    vegetarian: List vegetarian food only
        timezone: The timezone of your location in format of +/- hhmm
        from: The offset of items to be ignored in response for paging
        size: Number of items returned per response
        
    """
    url = f"https://tasty.p.rapidapi.com/feeds/list"
    querystring = {'vegetarian': vegetarian, 'timezone': timezone, 'from': is_from, 'size': size, }
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "tasty.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def feeds_list(vegetarian: bool, timezone: str, is_from: int, size: int, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "List latest feeds about new food, recipes,etc..."
    vegetarian: List vegetarian food only
        timezone: The timezone of your location in format of +/- hhmm
        from: The offset of items to be ignored in response for paging
        size: Number of items returned per response
        
    """
    url = f"https://tasty.p.rapidapi.com/feeds/list"
    querystring = {'vegetarian': vegetarian, 'timezone': timezone, 'from': is_from, 'size': size, }
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "tasty.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def tags_list(toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "List supported tags name for filtering in recipes/list API"
    
    """
    url = f"https://tasty.p.rapidapi.com/tags/list"
    querystring = {}
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "tasty.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def tags_list(toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "List supported tags name for filtering in recipes/list API"
    
    """
    url = f"https://tasty.p.rapidapi.com/tags/list"
    querystring = {}
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "tasty.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def recipes_list_similarities(recipe_id: int, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "List similar recipes by specific recipe id"
    recipe_id: The id value of any recipe returned in recipes/list API
        
    """
    url = f"https://tasty.p.rapidapi.com/recipes/list-similarities"
    querystring = {'recipe_id': recipe_id, }
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "tasty.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def recipes_auto_complete(prefix: str, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Get auto complete suggestions by name or ingredients"
    prefix: Food name or ingredient
        
    """
    url = f"https://tasty.p.rapidapi.com/recipes/auto-complete"
    querystring = {'prefix': prefix, }
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "tasty.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def recipes_get_more_info(is_id: int, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Get  more information of recipe if available, such as : ingredients, nutrition info, preparation, etc... This endpoint returns 404 status code, it means there is no more information to obtain.
		* .../recipes/list already returns ingredients, nutrition info, preparation, etc..."
    id: The id value of any recipe returned in recipes/list API
        
    """
    url = f"https://tasty.p.rapidapi.com/recipes/get-more-info"
    querystring = {'id': is_id, }
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "tasty.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

