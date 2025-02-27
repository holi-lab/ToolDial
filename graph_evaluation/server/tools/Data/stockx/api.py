import requests
import json
from datetime import date, datetime, timedelta
import os

from typing import Optional, Dict, Union, List


def product(query: str, currency: str='EUR', country: str='FR', toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Product informations"
    
    """
    url = f"https://stockx1.p.rapidapi.com/v2/stockx/product"
    querystring = {'query': query, }
    if currency:
        querystring['currency'] = currency
    if country:
        querystring['country'] = country
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "stockx1.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def trends(query: str, currency: str='EUR', toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "See the latest trends of Stockx"
    query: `['sneakers', 'streetwear', 'electronics', 'trading cards', 'collectibles', 'handbags', 'watches']`
        
    """
    url = f"https://stockx1.p.rapidapi.com/v2/stockx/trends"
    querystring = {'query': query, }
    if currency:
        querystring['currency'] = currency
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "stockx1.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def product_activity(type: str, query: str, currency: str='EUR', country: str='FR', toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "See the latest activities of an StockX ID"
    
    """
    url = f"https://stockx1.p.rapidapi.com/v2/stockx/product/activity"
    querystring = {'type': type, 'query': query, }
    if currency:
        querystring['currency'] = currency
    if country:
        querystring['country'] = country
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "stockx1.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def search(query: str, page: int=1, limit: int=10, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Search a reference name or SKU"
    
    """
    url = f"https://stockx1.p.rapidapi.com/v2/stockx/search"
    querystring = {'query': query, }
    if page:
        querystring['page'] = page
    if limit:
        querystring['limit'] = limit
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "stockx1.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def product_360(query: str, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Product 360images"
    
    """
    url = f"https://stockx1.p.rapidapi.com/v2/stockx/product/360"
    querystring = {'query': query, }
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "stockx1.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def product_chart(query: str, end_date: str, start_date: str, currency: str='EUR', toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "See the latest activities with graph of an StockX ID"
    
    """
    url = f"https://stockx1.p.rapidapi.com/v2/stockx/product/graph"
    querystring = {'query': query, 'end_date': end_date, 'start_date': start_date, }
    if currency:
        querystring['currency'] = currency
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "stockx1.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

