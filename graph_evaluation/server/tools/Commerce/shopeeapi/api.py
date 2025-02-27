import requests
import json
from datetime import date, datetime, timedelta
import os

from typing import Optional, Dict, Union, List


def get_products_from_sellerid_and_categoryid(sellerid: str, region: str, offset: int=0, limit: int=30, categoryid: str='11043778', toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Getting products from given sellerID and categoryID.
		
		If the categoryID is not provided it will fetch from any category from sellerID"
    sellerid: Required sellerID
        categoryid: An optional categoryID
        
    """
    url = f"https://shopeeapi2.p.rapidapi.com/{region}/seller/{sellerid}/products/{categoryid}"
    querystring = {}
    if offset:
        querystring['offset'] = offset
    if limit:
        querystring['limit'] = limit
    if categoryid:
        querystring['categoryID'] = categoryid
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "shopeeapi2.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def get_categories_from_sellerid(region: str, sellerid: str, limit: str='30', offset: int=0, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Getting product categories from seller ID"
    region: An region
        
    """
    url = f"https://shopeeapi2.p.rapidapi.com/{region}/seller/{sellerid}/categories"
    querystring = {}
    if limit:
        querystring['limit'] = limit
    if offset:
        querystring['offset'] = offset
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "shopeeapi2.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def search_products(region: str, q: str, sellerid: int=None, p: int=1, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Search product & Paginate"
    region: The region must one of [\\\\\\\"en\\\\\\\", \\\\\\\"sg\\\\\\\", \\\\\\\"my\\\\\\\", \\\\\\\"id\\\\\\\", \\\\\\\"th\\\\\\\", \\\\\\\"vn\\\\\\\", \\\\\\\"ph\\\\\\\", \\\\\\\"tw\\\\\\\", \\\\\\\"br\\\\\\\", \\\\\\\"cl\\\\\\\", \\\\\\\"mx\\\\\\\", \\\\\\\"co\\\\\\\"]
        q: A product search query
        sellerid: (Optional) Filter search product by seller/shop ID
        p: Page number
        
    """
    url = f"https://shopeeapi2.p.rapidapi.com/{region}/search"
    querystring = {'q': q, }
    if sellerid:
        querystring['sellerID'] = sellerid
    if p:
        querystring['p'] = p
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "shopeeapi2.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def get_product_details(region: str, path: str, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Get shopee product details"
    region: The region. must be one of [\\\\\\\"en\\\\\\\", \\\\\\\"sg\\\\\\\", \\\\\\\"my\\\\\\\", \\\\\\\"id\\\\\\\", \\\\\\\"th\\\\\\\", \\\\\\\"vn\\\\\\\", \\\\\\\"ph\\\\\\\", \\\\\\\"tw\\\\\\\", \\\\\\\"br\\\\\\\", \\\\\\\"cl\\\\\\\", \\\\\\\"mx\\\\\\\", \\\\\\\"co\\\\\\\"]
        path: Path parameters
        
    """
    url = f"https://shopeeapi2.p.rapidapi.com/{region}/{path}"
    querystring = {}
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "shopeeapi2.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

