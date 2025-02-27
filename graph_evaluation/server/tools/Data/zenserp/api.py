import requests
import json
from datetime import date, datetime, timedelta
import os

from typing import Optional, Dict, Union, List


def search(q: str, device: str=None, tbm: str=None, location: str='United States', search_engine: str='google.com', num: str=None, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Get a SERP"
    q: Query String (keyword)
        device: Which device to use: ['desktop', 'mobile']
        tbm: Set to 'isch' for image results
        location: location for the search engine
        search_engine: The url of the search engine to query
        
    """
    url = f"https://zenserp.p.rapidapi.com/search"
    querystring = {'q': q, }
    if device:
        querystring['device'] = device
    if tbm:
        querystring['tbm'] = tbm
    if location:
        querystring['location'] = location
    if search_engine:
        querystring['search_engine'] = search_engine
    if num:
        querystring['num'] = num
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "zenserp.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

