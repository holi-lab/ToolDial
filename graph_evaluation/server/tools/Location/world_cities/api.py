import requests
import json
from datetime import date, datetime, timedelta
import os

from typing import Optional, Dict, Union, List


def query(searchby: str='city', query: str='paris', toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Use a query with at least 3 characters to search for a city, state or country matching the query"
    searchby: Optionally search by city, state or country to narrow the results.
        query: It can be part of the name of a city, state or country. At least 3 characters are needed.
        
    """
    url = f"https://andruxnet-world-cities-v1.p.rapidapi.com/"
    querystring = {}
    if searchby:
        querystring['searchby'] = searchby
    if query:
        querystring['query'] = query
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "andruxnet-world-cities-v1.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

