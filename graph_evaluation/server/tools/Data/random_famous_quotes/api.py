import requests
import json
from datetime import date, datetime, timedelta
import os

from typing import Optional, Dict, Union, List


def get_endpoint(cat: str='famous', count: int=10, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Get endpoint for quotes by famous people or from popular movies"
    cat: Get quotes inside any of these categories:  "movies" or "famous". If not present get quotes from any category.
        count: Number of quotes to return. If omitted, a single quote will be returned. Max number of quotes is 10
        
    """
    url = f"https://andruxnet-random-famous-quotes.p.rapidapi.com/"
    querystring = {}
    if cat:
        querystring['cat'] = cat
    if count:
        querystring['count'] = count
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "andruxnet-random-famous-quotes.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

