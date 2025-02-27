import requests
import json
from datetime import date, datetime, timedelta
import os

from typing import Optional, Dict, Union, List


def name_generate(category: str, variation: str=None, suggest: str=None, limit: int=None, start: int=None, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Generated names in the given category"
    category: Category to generator names from
        variation: Variation if supported ( male/female/any )
        suggest: Suggestion string if supported by this category generator.
        limit: limit. Controls pagination limit. Relevant only if suggestion is supported
        start: start. Controls pagination. Relevant only if suggestion is supported
        
    """
    url = f"https://name-generation.p.rapidapi.com/name/generate"
    querystring = {'category': category, }
    if variation:
        querystring['variation'] = variation
    if suggest:
        querystring['suggest'] = suggest
    if limit:
        querystring['limit'] = limit
    if start:
        querystring['start'] = start
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "name-generation.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def name_categories(start: int=None, limit: int=None, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Get available name generation categories."
    start: start
        limit: limit
        
    """
    url = f"https://name-generation.p.rapidapi.com/name/categories"
    querystring = {}
    if start:
        querystring['start'] = start
    if limit:
        querystring['limit'] = limit
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "name-generation.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

