import requests
import json
from datetime import date, datetime, timedelta
import os

from typing import Optional, Dict, Union, List


def hot_products(search_value: str, max_price: int=10000, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Get hot products by search value, you can set maximum search price too. You will get up to 50 results."
    
    """
    url = f"https://aliexpress-true-api.p.rapidapi.com/hot_products/{search_value}"
    querystring = {}
    if max_price:
        querystring['max_price'] = max_price
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "aliexpress-true-api.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def product_by_id(product_id: int, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Get entirely a single product and all possible values comes from it including images, videos and all product data."
    
    """
    url = f"https://aliexpress-true-api.p.rapidapi.com/product/{product_id}"
    querystring = {}
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "aliexpress-true-api.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

