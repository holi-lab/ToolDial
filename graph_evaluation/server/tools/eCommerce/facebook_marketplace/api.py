import requests
import json
from datetime import date, datetime, timedelta
import os

from typing import Optional, Dict, Union, List


def categories(toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Facebook items categories"
    
    """
    url = f"https://facebook-marketplace.p.rapidapi.com/categories"
    querystring = {}
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "facebook-marketplace.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def products(no_of_save: int=25, min_sales: int=5, min_price: str='2.00', category: str='Apparel', city: str='New York', page: int=1, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Get facbook items"
    
    """
    url = f"https://facebook-marketplace.p.rapidapi.com/products"
    querystring = {}
    if no_of_save:
        querystring['no_of_save'] = no_of_save
    if min_sales:
        querystring['min_sales'] = min_sales
    if min_price:
        querystring['min_price'] = min_price
    if category:
        querystring['category'] = category
    if city:
        querystring['city'] = city
    if page:
        querystring['page'] = page
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "facebook-marketplace.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def cities(toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Facebook items cities"
    
    """
    url = f"https://facebook-marketplace.p.rapidapi.com/cities"
    querystring = {}
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "facebook-marketplace.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

