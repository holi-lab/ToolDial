import requests
import json
from datetime import date, datetime, timedelta
import os

from typing import Optional, Dict, Union, List


def get_search_results(query: str, limit: int=10, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Insert the search query and retrieve the most relevant results from DuckDuckGo Search in JSON format."
    query: The search string of your choice.
        limit: The number of results you want to retrieve.
        
    """
    url = f"https://duck-duck-go-search-api.p.rapidapi.com/duckduckgo"
    querystring = {'query': query, }
    if limit:
        querystring['limit'] = limit
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "duck-duck-go-search-api.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

