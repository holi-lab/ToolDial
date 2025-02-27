import requests
import json
from datetime import date, datetime, timedelta
import os

from typing import Optional, Dict, Union, List


def recipe_search_and_recommendations(q: str='chicken', r: str=None, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Search 2 million recipes using keywords, 28 nutrients and 40 diet and health filters"
    
    """
    url = f"https://edamam-recipe-search.p.rapidapi.com/search"
    querystring = {}
    if q:
        querystring['q'] = q
    if r:
        querystring['r'] = r
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "edamam-recipe-search.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

