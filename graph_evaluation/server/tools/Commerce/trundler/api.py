import requests
import json
from datetime import date, datetime, timedelta
import os

from typing import Optional, Dict, Union, List


def get_product_detail(is_id: int, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Details of a specific product."
    id: A product ID
        
    """
    url = f"https://trundler.p.rapidapi.com/product/{is_id}"
    querystring = {}
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "trundler.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def get_stats(toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    " "
    
    """
    url = f"https://trundler.p.rapidapi.com/admin/stats"
    querystring = {}
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "trundler.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def get_stats_status(toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    " "
    
    """
    url = f"https://trundler.p.rapidapi.com/admin/stats/status"
    querystring = {}
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "trundler.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def get_auth(toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    " "
    
    """
    url = f"https://trundler.p.rapidapi.com/admin/auth"
    querystring = {}
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "trundler.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def get_category_products(is_id: int, limit: int=100, recursive: bool=None, offset: int=None, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Products in a specific category."
    id: A category ID
        limit: Limit the number of items
        recursive: All categories in sub-tree
        offset: Number of items to skip
        
    """
    url = f"https://trundler.p.rapidapi.com/category/{is_id}"
    querystring = {}
    if limit:
        querystring['limit'] = limit
    if recursive:
        querystring['recursive'] = recursive
    if offset:
        querystring['offset'] = offset
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "trundler.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def get_retailer(toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "List all retailers."
    
    """
    url = f"https://trundler.p.rapidapi.com/retailer"
    querystring = {}
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "trundler.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def get_category(apex: int=None, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "List all categories."
    apex: A category ID at which to begin tree
        
    """
    url = f"https://trundler.p.rapidapi.com/category"
    querystring = {}
    if apex:
        querystring['apex'] = apex
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "trundler.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def get_retailer_detail(is_id: int, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Details of a specific retailer."
    id: A retailer ID
        
    """
    url = f"https://trundler.p.rapidapi.com/retailer/{is_id}"
    querystring = {}
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "trundler.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def get_stats_daily(toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    " "
    
    """
    url = f"https://trundler.p.rapidapi.com/admin/stats/daily"
    querystring = {}
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "trundler.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def get_auth_count(toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    " "
    
    """
    url = f"https://trundler.p.rapidapi.com/admin/auth/count"
    querystring = {}
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "trundler.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def get_product_prices(is_id: int, limit: int=100, offset: int=None, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Price history of a specific product."
    id: A product ID
        limit: Limit the number of items
        offset: Number of items to skip
        
    """
    url = f"https://trundler.p.rapidapi.com/product/{is_id}/price"
    querystring = {}
    if limit:
        querystring['limit'] = limit
    if offset:
        querystring['offset'] = offset
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "trundler.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def get_product(offset: int=None, regex: bool=None, ignore_case: bool=None, barcode: str='5011013506132', product: str='coffee', limit: int=100, brand: str='Nescaf[eé]', toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "List all products."
    offset: Number of items to skip
        regex: Filters are REGEX
        ignore_case: Filters ignore case
        barcode: Barcode filter
        product: Product name filter
        limit: Limit the number of items
        brand: Product brand filter
        
    """
    url = f"https://trundler.p.rapidapi.com/product"
    querystring = {}
    if offset:
        querystring['offset'] = offset
    if regex:
        querystring['regex'] = regex
    if ignore_case:
        querystring['ignore_case'] = ignore_case
    if barcode:
        querystring['barcode'] = barcode
    if product:
        querystring['product'] = product
    if limit:
        querystring['limit'] = limit
    if brand:
        querystring['brand'] = brand
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "trundler.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def get_retailer_products(is_id: int, product: str=None, ignore_case: bool=None, regex: bool=None, brand: str=None, limit: int=100, offset: int=None, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Products for a specific retailer. Can filter by product or brand name using REGEX."
    id: A retailer ID
        product: Product name filter
        ignore_case: Filters ignore case
        regex: Filters are REGEX
        brand: Product brand filter
        limit: Limit the number of items
        offset: Number of items to skip
        
    """
    url = f"https://trundler.p.rapidapi.com/retailer/{is_id}/product"
    querystring = {}
    if product:
        querystring['product'] = product
    if ignore_case:
        querystring['ignore_case'] = ignore_case
    if regex:
        querystring['regex'] = regex
    if brand:
        querystring['brand'] = brand
    if limit:
        querystring['limit'] = limit
    if offset:
        querystring['offset'] = offset
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "trundler.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

