import requests
import json
from datetime import date, datetime, timedelta
import os

from typing import Optional, Dict, Union, List


def categories_list(category: str=None, limit: int=None, offset: int=None, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    ""
    category: filter by category and subcategory
        limit: pagination limit
        offset: pagination offset
        
    """
    url = f"https://xco-poc.p.rapidapi.com/categories/list"
    querystring = {}
    if category:
        querystring['category'] = category
    if limit:
        querystring['limit'] = limit
    if offset:
        querystring['offset'] = offset
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "xco-poc.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def products(limit: int=None, category: str=None, sort: str='sales', offset: int=None, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    ""
    limit: pagination limit
        category: Filter products by category
        sort: Apply a sort on the result
        offset: pagination offset
        
    """
    url = f"https://xco-poc.p.rapidapi.com/products"
    querystring = {}
    if limit:
        querystring['limit'] = limit
    if category:
        querystring['category'] = category
    if sort:
        querystring['sort'] = sort
    if offset:
        querystring['offset'] = offset
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "xco-poc.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

