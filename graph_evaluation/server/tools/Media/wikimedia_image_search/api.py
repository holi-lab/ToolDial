import requests
import json
from datetime import date, datetime, timedelta
import os

from typing import Optional, Dict, Union, List


def image_search(query: str, results: int=12, page: int=1, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "wiki Image search"
    query: Query String for Search
        results: Number of Results
        page: Page 1,2,3,4,5 =means Offset: 12 24 36 48
        
    """
    url = f"https://wikimedia-image-search.p.rapidapi.com/wiki/"
    querystring = {'query': query, }
    if results:
        querystring['results'] = results
    if page:
        querystring['page'] = page
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "wikimedia-image-search.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

