import requests
import json
from datetime import date, datetime, timedelta
import os

from typing import Optional, Dict, Union, List


def fetch_shows(query: str='{ "title": "Serial" }', is_from: str='0', order: str=None, size: str='20', toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Get a list of podcast shows. You can optionally pass in a query object."
    
    """
    url = f"https://casterdata.p.rapidapi.com/api/1.0/shows"
    querystring = {}
    if query:
        querystring['query'] = query
    if is_from:
        querystring['from'] = is_from
    if order:
        querystring['order'] = order
    if size:
        querystring['size'] = size
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "casterdata.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

