import requests
import json
from datetime import date, datetime, timedelta
import os

from typing import Optional, Dict, Union, List


def search_product(keyword: str, shopnameid: str, extrameta: int=1, strategy: str=None, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Crawl live data from the webpage"
    keyword: Search key string
        shopnameid: NameId of the target shop
        extrameta: a flag used to add extrameta in the response
        strategy: speed up response time by activating cache
        
    """
    url = f"https://whatson-amazon.p.rapidapi.com/item/_search"
    querystring = {'keyword': keyword, 'shopNameId': shopnameid, }
    if extrameta:
        querystring['extrameta'] = extrameta
    if strategy:
        querystring['strategy'] = strategy
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "whatson-amazon.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def list_shops(page: int=0, size: int=30, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "List supported shops"
    page: Start from 0
        size: Size of the page, min : 30 - max : 50
        
    """
    url = f"https://whatson-amazon.p.rapidapi.com/shop/_list"
    querystring = {}
    if page:
        querystring['page'] = page
    if size:
        querystring['size'] = size
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "whatson-amazon.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

