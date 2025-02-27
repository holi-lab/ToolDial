import requests
import json
from datetime import date, datetime, timedelta
import os

from typing import Optional, Dict, Union, List


def farfetch_search_product(brand_id: str='117712', keyword: str='nanushka', page: str='1', sort: str='low_to_high', category_list_id: str='136227', toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Searching realtime data from Farfetch"
    sort: `our_pick , low_to_high , high_to_low , newest`
        category_list_id: Separate by comma 
Example : 136227,135981
        
    """
    url = f"https://farfetch-data.p.rapidapi.com/search.php"
    querystring = {}
    if brand_id:
        querystring['brand_id'] = brand_id
    if keyword:
        querystring['keyword'] = keyword
    if page:
        querystring['page'] = page
    if sort:
        querystring['sort'] = sort
    if category_list_id:
        querystring['category_list_id'] = category_list_id
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "farfetch-data.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def farfetch_lookup_by_product_id(product_id: str, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Get realtime product information by product_id from Farfetch"
    
    """
    url = f"https://farfetch-data.p.rapidapi.com/product.php"
    querystring = {'product_id': product_id, }
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "farfetch-data.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def farfetch_categories(toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Realtime categories from Farfetch"
    
    """
    url = f"https://farfetch-data.p.rapidapi.com/category.php"
    querystring = {}
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "farfetch-data.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

