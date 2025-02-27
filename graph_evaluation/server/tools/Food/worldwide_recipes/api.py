import requests
import json
from datetime import date, datetime, timedelta
import os

from typing import Optional, Dict, Union, List


def recipes_by_author(profile_name: str, q: str=None, start: int=None, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Get recipes by author"
    start: For pagination, eg: 

- Page 1 = start:0
- Page 2 = start:20
- Page 3 = start:40

and so on...
        
    """
    url = f"https://worldwide-recipes1.p.rapidapi.com/api/recipes-by-author"
    querystring = {'profile_name': profile_name, }
    if q:
        querystring['q'] = q
    if start:
        querystring['start'] = start
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "worldwide-recipes1.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def more_from_author(profile_display_name: str, canonical_term: str, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Get more recipe from author"
    
    """
    url = f"https://worldwide-recipes1.p.rapidapi.com/api/more-from-author"
    querystring = {'profile_display_name': profile_display_name, 'canonical_term': canonical_term, }
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "worldwide-recipes1.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def related(related_product_id: str, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Get related recipes"
    
    """
    url = f"https://worldwide-recipes1.p.rapidapi.com/api/related"
    querystring = {'related_product_id': related_product_id, }
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "worldwide-recipes1.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def review(recipe_id: str, offset: int=None, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Get reviews"
    offset: For pagination
        
    """
    url = f"https://worldwide-recipes1.p.rapidapi.com/api/review"
    querystring = {'recipe_id': recipe_id, }
    if offset:
        querystring['offset'] = offset
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "worldwide-recipes1.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def detail(canonical_term: str, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Get detail of recipe"
    
    """
    url = f"https://worldwide-recipes1.p.rapidapi.com/api/detail"
    querystring = {'canonical_term': canonical_term, }
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "worldwide-recipes1.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def search(q: str, start: int=None, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Search Recipe"
    start: For pagination, eg: 

- Page 1 = start:0
- Page 2 = start:20
- Page 3 = start:40

and so on...
        
    """
    url = f"https://worldwide-recipes1.p.rapidapi.com/api/search"
    querystring = {'q': q, }
    if start:
        querystring['start'] = start
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "worldwide-recipes1.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def explore(start: int=None, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Explore Recipes"
    start: For pagination, eg: 

- Page 1 = start:0
- Page 2 = start:20
- Page 3 = start:40

and so on...
        
    """
    url = f"https://worldwide-recipes1.p.rapidapi.com/api/explore"
    querystring = {}
    if start:
        querystring['start'] = start
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "worldwide-recipes1.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def suggestions(q: str, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Get Suggestions"
    
    """
    url = f"https://worldwide-recipes1.p.rapidapi.com/api/suggestions"
    querystring = {'q': q, }
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "worldwide-recipes1.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

