import requests
import json
from datetime import date, datetime, timedelta
import os

from typing import Optional, Dict, Union, List


def batch(query: str=None, show_query: int=0, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "API endpoint used to perform batch language detection"
    query: Append multiple "query[]" parameters containing query texts
        show_query: Set to "1" if you want the API's JSON result set to return your query text.
        
    """
    url = f"https://apilayer-languagelayer-v1.p.rapidapi.com/batch"
    querystring = {}
    if query:
        querystring['query[]'] = query
    if show_query:
        querystring['show_query'] = show_query
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "apilayer-languagelayer-v1.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

