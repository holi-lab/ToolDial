import requests
import json
from datetime import date, datetime, timedelta
import os

from typing import Optional, Dict, Union, List


def single_cannabis_recipe_by_id(authorization: str, is_id: int, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Get all information on the single cannabis recipe by ID."
    
    """
    url = f"https://cannabis-recipes.p.rapidapi.com/recipes/"
    querystring = {'Authorization': authorization, 'id': is_id, }
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "cannabis-recipes.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def search_cannabis_recipes(authorization: str, search: str, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Search cannabis recipes by keyword"
    
    """
    url = f"https://cannabis-recipes.p.rapidapi.com/recipes/"
    querystring = {'Authorization': authorization, 'search': search, }
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "cannabis-recipes.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def all_cannabis_recipes_paginated(authorization: str, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Get all cannabis recipes but in a set of 10 with pagination. Use ?page= to change the page and go to the next page."
    
    """
    url = f"https://cannabis-recipes.p.rapidapi.com/paginated_recipes/"
    querystring = {'Authorization': authorization, }
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "cannabis-recipes.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def all_cannabis_recipes(authorization: str, au: str=None, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Get all cannabis recipes"
    
    """
    url = f"https://cannabis-recipes.p.rapidapi.com/recipes/"
    querystring = {'Authorization': authorization, }
    if au:
        querystring['au'] = au
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "cannabis-recipes.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def search_tags(authorization: str, tags_icontains: str, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Search cannabis recipes by tags"
    
    """
    url = f"https://cannabis-recipes.p.rapidapi.com/recipes/"
    querystring = {'Authorization': authorization, 'tags__icontains': tags_icontains, }
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "cannabis-recipes.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def cannabis_recipes_by_category_id(authorization: str, category: int, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Get all cannabis recipes by category."
    
    """
    url = f"https://cannabis-recipes.p.rapidapi.com/recipes/"
    querystring = {'Authorization': authorization, 'category': category, }
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "cannabis-recipes.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def categories(authorization: str, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Get all categories"
    
    """
    url = f"https://cannabis-recipes.p.rapidapi.com/categories/"
    querystring = {'Authorization': authorization, }
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "cannabis-recipes.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

