import requests
import json
from datetime import date, datetime, timedelta
import os

from typing import Optional, Dict, Union, List


def v1_recipe(query: str, offset: str=None, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Get a list of recipes for a given search query. Returns at most 10 results. To access more than the first 10 results, use the offset parameter to offset results in multiple API calls."
    query: query text to search.
        offset: number of results to offset for pagination.
        
    """
    url = f"https://recipe-by-api-ninjas.p.rapidapi.com/v1/recipe"
    querystring = {'query': query, }
    if offset:
        querystring['offset'] = offset
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "recipe-by-api-ninjas.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

