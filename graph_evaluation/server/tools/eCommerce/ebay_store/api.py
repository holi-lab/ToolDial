import requests
import json
from datetime import date, datetime, timedelta
import os

from typing import Optional, Dict, Union, List


def search_items(kw: str='jeans', high: str='99.5', shipping: bool=None, location: int=0, limit: int=5, low: int=5, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Search items on eBay with given **keyword**, **Price range** and **shipping charges**."
    
    """
    url = f"https://ebay-store.p.rapidapi.com/api/search"
    querystring = {}
    if kw:
        querystring['kw'] = kw
    if high:
        querystring['high'] = high
    if shipping:
        querystring['shipping'] = shipping
    if location:
        querystring['location'] = location
    if limit:
        querystring['limit'] = limit
    if low:
        querystring['low'] = low
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "ebay-store.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

