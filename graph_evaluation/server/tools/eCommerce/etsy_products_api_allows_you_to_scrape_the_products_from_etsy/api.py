import requests
import json
from datetime import date, datetime, timedelta
import os

from typing import Optional, Dict, Union, List


def getproducts(page: int=None, query: str='iphone 14', period: int=None, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Get Etsy products"
    
    """
    url = f"https://etsy-products-api-allows-you-to-scrape-the-products-from-etsy.p.rapidapi.com/"
    querystring = {}
    if page:
        querystring['page'] = page
    if query:
        querystring['query'] = query
    if period:
        querystring['period'] = period
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "etsy-products-api-allows-you-to-scrape-the-products-from-etsy.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

