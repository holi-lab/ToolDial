import requests
import json
from datetime import date, datetime, timedelta
import os

from typing import Optional, Dict, Union, List


def zappos_product_detail_by_product_id(product_id: int, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Get realtime product information by product_id from Zappos"
    
    """
    url = f"https://zappos-realtime-data.p.rapidapi.com/product.php"
    querystring = {'product_id': product_id, }
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "zappos-realtime-data.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def zappos_search_product(keyword: str, sort: str='new_arrivals', page: int=1, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Searching realtime data from Zappos"
    sort: `best_seller , relevance, low_to_high, high_to_low, customer_rating, new_arrivals, brand_name`
        
    """
    url = f"https://zappos-realtime-data.p.rapidapi.com/search.php"
    querystring = {'keyword': keyword, }
    if sort:
        querystring['sort'] = sort
    if page:
        querystring['page'] = page
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "zappos-realtime-data.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def category_list(toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Realtime categories from Zappos"
    
    """
    url = f"https://zappos-realtime-data.p.rapidapi.com/category.php"
    querystring = {}
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "zappos-realtime-data.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

