import requests
import json
from datetime import date, datetime, timedelta
import os

from typing import Optional, Dict, Union, List


def get_character_by_id(is_id: int, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Get individual character by ID
		Options:
		
		- Limit → Limit amount of responses received
		- Step → Skip amount of characters"
    
    """
    url = f"https://riordanverse-api.p.rapidapi.com/api/characters/{is_id}"
    querystring = {}
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "riordanverse-api.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def get_all_characters(limit: int=None, search: str=None, skip: int=None, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Get all characters
		
		Options:
		
		- Limit → Limit amount of responses received
		- Step → Skip amount of characters
		- Search → Return characters with provided substring"
    
    """
    url = f"https://riordanverse-api.p.rapidapi.com/api/characters"
    querystring = {}
    if limit:
        querystring['limit'] = limit
    if search:
        querystring['search'] = search
    if skip:
        querystring['skip'] = skip
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "riordanverse-api.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

