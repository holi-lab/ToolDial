import requests
import json
from datetime import date, datetime, timedelta
import os

from typing import Optional, Dict, Union, List


def search_products(keyword: str, lan: str=None, page: int=1, pagesize: int=10, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "search products
		The source of the data can be controlled through the 'lan' field (en: English website, cn: Chinese website)"
    
    """
    url = f"https://yiwugo-product-data.p.rapidapi.com/api/sc/yiwugo/search/product"
    querystring = {'keyword': keyword, }
    if lan:
        querystring['lan'] = lan
    if page:
        querystring['page'] = page
    if pagesize:
        querystring['pageSize'] = pagesize
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "yiwugo-product-data.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def search_shops(keyword: str, lan: str=None, page: int=1, pagesize: int=10, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "search shops by keyword.
		The source of the data can be controlled through the 'lan' field (en: English website, cn: Chinese website)"
    
    """
    url = f"https://yiwugo-product-data.p.rapidapi.com/api/sc/yiwugo/search/shop"
    querystring = {'keyword': keyword, }
    if lan:
        querystring['lan'] = lan
    if page:
        querystring['page'] = page
    if pagesize:
        querystring['pageSize'] = pagesize
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "yiwugo-product-data.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def get_product_detail(item_id: int, lan: str=None, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "get product detail by item_id,
		The source of the data can be controlled through the 'lan' field (en: English website, cn: Chinese website)"
    
    """
    url = f"https://yiwugo-product-data.p.rapidapi.com/api/sc/yiwugo/item_detail"
    querystring = {'item_id': item_id, }
    if lan:
        querystring['lan'] = lan
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "yiwugo-product-data.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

