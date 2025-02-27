import requests
import json
from datetime import date, datetime, timedelta
import os

from typing import Optional, Dict, Union, List


def countries(toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Get countries from ASOS"
    
    """
    url = f"https://asos-api1.p.rapidapi.com/countries"
    querystring = {}
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "asos-api1.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def categories(toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Get categories and menus from ASOS"
    
    """
    url = f"https://asos-api1.p.rapidapi.com/categories"
    querystring = {}
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "asos-api1.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def products(limit: int, offset: int, categoryid: int, store: str, lang: str='en-GB', currency: str='GBP', sort: str='freshness', country: str='KZ', sizeschema: str='US', toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Get products list"
    limit: The number of items per page
        offset: The offset to skip already viewed products
        categoryid: The value of categoryId field from categories API
        store: Get value from store field from countries API
        lang: The language code
        currency: Currency from countries API
        sort: One of the following pricedesc|priceasc|freshness (Newest)
        country: The country code
        sizeschema: Get suitable value from countries API
        
    """
    url = f"https://asos-api1.p.rapidapi.com/products"
    querystring = {'limit': limit, 'offset': offset, 'categoryId': categoryid, 'store': store, }
    if lang:
        querystring['lang'] = lang
    if currency:
        querystring['currency'] = currency
    if sort:
        querystring['sort'] = sort
    if country:
        querystring['country'] = country
    if sizeschema:
        querystring['sizeSchema'] = sizeschema
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "asos-api1.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def product(is_id: int, lang: str='en-US', store: str='US', currency: str='USD', sizeschema: str='US', toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Get product with id"
    lang: Get suitable value from countries API
        store: Get suitable store value from countries API, this param is not sideId as in products API
        currency: Get suitable value from countries API
        sizeschema: Get suitable value from countries API
        
    """
    url = f"https://asos-api1.p.rapidapi.com/products"
    querystring = {'id': is_id, }
    if lang:
        querystring['lang'] = lang
    if store:
        querystring['store'] = store
    if currency:
        querystring['currency'] = currency
    if sizeschema:
        querystring['sizeSchema'] = sizeschema
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "asos-api1.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

