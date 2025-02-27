import requests
import json
from datetime import date, datetime, timedelta
import os

from typing import Optional, Dict, Union, List


def taunt_generate(category: str, limit: int=None, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Generated taunts in the given category"
    category: Category to generator taunt from
        limit: Limit. Controls number of taunts generated. Max of 5-10 based on the plan
        
    """
    url = f"https://insult-generation.p.rapidapi.com/taunt/generate"
    querystring = {'category': category, }
    if limit:
        querystring['limit'] = limit
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "insult-generation.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def taunt_categories(start: int=None, limit: int=None, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Get available taunt generation categories."
    start: start
        limit: limit
        
    """
    url = f"https://insult-generation.p.rapidapi.com/taunt/categories"
    querystring = {}
    if start:
        querystring['start'] = start
    if limit:
        querystring['limit'] = limit
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "insult-generation.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

