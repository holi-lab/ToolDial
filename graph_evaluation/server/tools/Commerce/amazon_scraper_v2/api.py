import requests
import json
from datetime import date, datetime, timedelta
import os

from typing import Optional, Dict, Union, List


def get_product(asin: str, country: str='GB', currency: str='GBP', toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Get Product"
    
    """
    url = f"https://amazon-scraper32.p.rapidapi.com/get-product"
    querystring = {'asin': asin, }
    if country:
        querystring['country'] = country
    if currency:
        querystring['currency'] = currency
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "amazon-scraper32.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def search_product(query: str, sort: str=None, country: str='GB', currency: str='GBP', page: str='1', toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Search Product"
    
    """
    url = f"https://amazon-scraper32.p.rapidapi.com/search-product"
    querystring = {'query': query, }
    if sort:
        querystring['sort'] = sort
    if country:
        querystring['country'] = country
    if currency:
        querystring['currency'] = currency
    if page:
        querystring['page'] = page
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "amazon-scraper32.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

