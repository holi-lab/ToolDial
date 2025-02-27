import requests
import json
from datetime import date, datetime, timedelta
import os

from typing import Optional, Dict, Union, List


def product(is_id: str, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    ""
    id: Product ID
        
    """
    url = f"https://shopello.p.rapidapi.com/1/products/{is_id}.json"
    querystring = {}
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "shopello.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def stores(toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    ""
    
    """
    url = f"https://shopello.p.rapidapi.com/1/stores.json"
    querystring = {}
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "shopello.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def category(is_id: str, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    ""
    
    """
    url = f"https://shopello.p.rapidapi.com/1/categories/{is_id}.json"
    querystring = {}
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "shopello.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def categories(toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    ""
    
    """
    url = f"https://shopello.p.rapidapi.com/1/categories.json"
    querystring = {}
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "shopello.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def store(is_id: str, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    ""
    
    """
    url = f"https://shopello.p.rapidapi.com/1/stores/{is_id}.json"
    querystring = {}
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "shopello.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def related_products(is_id: str, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    ""
    id: Product ID
        
    """
    url = f"https://shopello.p.rapidapi.com/1/related_products/{is_id}.json"
    querystring = {}
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "shopello.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def products(store_id: str=None, category_id: str=None, query: str=None, limit: int=None, offset: str=None, price_min: int=None, price_max: int=None, has_images: int=None, order_by: str=None, order: str=None, fields: str=None, group_by: str=None, mode: str=None, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    ""
    store_id: One or more store_id's
        category_id: One or more category_id's
        query: Search query
        limit: 0 - 1000
        offset: 0 - 1000
        has_images: 0 or 1
        order_by: price, name, clicks, popular or relevance
        order: ASC or DESC
        fields: Which fields to fetch, comma separated
        mode: When passed along it will match each word in the query
        
    """
    url = f"https://shopello.p.rapidapi.com/1/products.json"
    querystring = {}
    if store_id:
        querystring['store_id'] = store_id
    if category_id:
        querystring['category_id'] = category_id
    if query:
        querystring['query'] = query
    if limit:
        querystring['limit'] = limit
    if offset:
        querystring['offset'] = offset
    if price_min:
        querystring['price_min'] = price_min
    if price_max:
        querystring['price_max'] = price_max
    if has_images:
        querystring['has_images'] = has_images
    if order_by:
        querystring['order_by'] = order_by
    if order:
        querystring['order'] = order
    if fields:
        querystring['fields'] = fields
    if group_by:
        querystring['group_by'] = group_by
    if mode:
        querystring['mode'] = mode
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "shopello.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

