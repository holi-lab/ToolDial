import requests
import json
from datetime import date, datetime, timedelta
import os

from typing import Optional, Dict, Union, List


def list_all_products_in_your_account(startfrom: int, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "This endpoint returns information about the products that have been added to Prisync via web UI or the API. The result is paginated. Each page contains up to 100 products."
    startfrom: Offset for pagination. It can take 0 and exact multiples of 100 as a value.
        
    """
    url = f"https://prisync.p.rapidapi.com/list/product/startFrom/{startfrom}"
    querystring = {}
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "prisync.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def get_a_specific_category(is_id: int, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "This endpoint returns category info with given id.  Tip You can get the id of a Category using /get/category/name/{name} request."
    id: Unique id of the category
        
    """
    url = f"https://prisync.p.rapidapi.com/get/category/id/{is_id}"
    querystring = {}
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "prisync.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def get_a_specific_url(is_id: int, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "This endpoint returns URL info with given id.   Tip: When you want to get all URLs belonging to a particular product, first you should make a request for getting the details of the product (See get product.). That request will return an array of URL id's in the response. Using those URL id's, you can get exact URL (the actual address) using this endpoint."
    id: Unique id of the URL
        
    """
    url = f"https://prisync.p.rapidapi.com/get/url/id/{is_id}"
    querystring = {}
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "prisync.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def list_all_products_in_your_account_with_price_information(startfrom: int, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "This endpoint returns information about the products including price information that have been added to Prisync via web UI or the API. The result is paginated. Each page contains up to 100 products."
    startfrom: Offset for pagination. It can take 0 and exact multiples of 100 as a value. 
        
    """
    url = f"https://prisync.p.rapidapi.com/list/product/summary/startFrom/{startfrom}"
    querystring = {}
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "prisync.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def get_a_particular_product(is_id: int, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "This endpoint returns product info with given id."
    id: Unique id of the product
        
    """
    url = f"https://prisync.p.rapidapi.com/get/product/id/{is_id}"
    querystring = {}
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "prisync.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def returns_account_info(toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "This endpoint returns your account details."
    
    """
    url = f"https://prisync.p.rapidapi.com/account"
    querystring = {}
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "prisync.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def list_all_brands_in_the_store(startfrom: int, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "This endpoint returns list of your brands. The response is paginated and it can show 100 brands at a time."
    startfrom: Offset for pagination. It can take 0 and exact multiples of 100 as a value.   P.S.: Actually, startFrom is an optional parameter. When not provided, this method returns the first page only. So that, calling without startFrom is equivalent to startFrom/0.
        
    """
    url = f"https://prisync.p.rapidapi.com/list/brand/startFrom/{startfrom}"
    querystring = {}
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "prisync.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def get_status_of_a_batch_import(toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "This endpoint returns progress status of batch import product.  /add/batch request works asynchronously which means it returns success immediately, then it adds all the products into a buffer queue, and finally it saves products into Prisync database. You can query this endpoint to get current status of your bulk add."
    
    """
    url = f"https://prisync.p.rapidapi.com/progress/batchImport"
    querystring = {}
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "prisync.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def list_all_categories_in_the_store(startfrom: int, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "This endpoint returns list of your categories. The response is paginated and it can show 100 categories at a time."
    startfrom: Offset for pagination. It can take 0 and exact multiples of 100 as a value.   P.S.: Actually, startFrom is an optional parameter. When not provided, this method returns the first page only. So that, calling without startFrom is equivalent to startFrom/0.
        
    """
    url = f"https://prisync.p.rapidapi.com/list/category/startFrom/{startfrom}"
    querystring = {}
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "prisync.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def get_a_specific_brand(is_id: int, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "This endpoint returns brand info with given id.  Tip You can get the id of a Brand using /get/brand/name/{name} request."
    id: Unique id of the brand
        
    """
    url = f"https://prisync.p.rapidapi.com/get/brand/id/{is_id}"
    querystring = {}
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "prisync.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

