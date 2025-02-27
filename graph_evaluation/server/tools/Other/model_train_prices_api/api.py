import requests
import json
from datetime import date, datetime, timedelta
import os

from typing import Optional, Dict, Union, List


def product_available(key: str=None, model: str=None, brand: str=None, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Check if a specific product is available or not."
    
    """
    url = f"https://model-train-prices-api.p.rapidapi.com/product-available"
    querystring = {}
    if key:
        querystring['key'] = key
    if model:
        querystring['model'] = model
    if brand:
        querystring['brand'] = brand
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "model-train-prices-api.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def get_product(key: str=None, model: str=None, currency: str=None, brand: str=None, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Get data about a specific product."
    
    """
    url = f"https://model-train-prices-api.p.rapidapi.com/get-product"
    querystring = {}
    if key:
        querystring['key'] = key
    if model:
        querystring['model'] = model
    if currency:
        querystring['currency'] = currency
    if brand:
        querystring['brand'] = brand
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "model-train-prices-api.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

