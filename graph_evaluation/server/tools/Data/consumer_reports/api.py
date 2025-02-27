import requests
import json
from datetime import date, datetime, timedelta
import os

from typing import Optional, Dict, Union, List


def brands_detail(is_id: int, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Get detailed information of a brand"
    id: The value of brands -> id field returned in .../search endpoint OR brandId field returned in .../products/list, .../products/detail endpoint.
        
    """
    url = f"https://consumer-reports.p.rapidapi.com/brands/detail"
    querystring = {'id': is_id, }
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "consumer-reports.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def products_get_offers(modelid: int, page: int=0, size: int=100, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Get offers from places or sites to buy product"
    modelid: The value of _id field returned in .../products/list endpoint
        page: The page index starting from 0, for paging purpose
        size: The number of items per response, for paging purpose. 
        
    """
    url = f"https://consumer-reports.p.rapidapi.com/products/get-offers"
    querystring = {'modelId': modelid, }
    if page:
        querystring['page'] = page
    if size:
        querystring['size'] = size
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "consumer-reports.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def products_detail(is_id: int, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Get detailed information of a product"
    id: The value of _id field returned in .../products/list endpoint
        
    """
    url = f"https://consumer-reports.p.rapidapi.com/products/detail"
    querystring = {'id': is_id, }
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "consumer-reports.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def products_list(productgroupid: int=None, size: int=100, page: int=0, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "List products from different categories"
    productgroupid: The value of _id field returned in .../product-groups/list endpoint
        size: The number of items per response, for paging purpose. Maximum is 250.
        page: The page index starting from 0, for paging purpose
        
    """
    url = f"https://consumer-reports.p.rapidapi.com/products/list"
    querystring = {}
    if productgroupid:
        querystring['productGroupId'] = productgroupid
    if size:
        querystring['size'] = size
    if page:
        querystring['page'] = page
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "consumer-reports.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def product_groups_detail(is_id: int, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Get detailed information of product group"
    id: The value of _id field returned in .../product-groups/list endpoint
        
    """
    url = f"https://consumer-reports.p.rapidapi.com/product-groups/detail"
    querystring = {'id': is_id, }
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "consumer-reports.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def product_groups_list(size: int=100, productgrouptypeid: int=None, page: int=None, parentproductgroupid: int=None, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "List product groups from categories and types"
    size: The number of items per response, for paging purpose. Maximum is 500.
        productgrouptypeid: The value of productGroupTypeId field returned right in this endpoint
        page: The page index starting from 0, for paging purpose
        parentproductgroupid: The value of _id field returned right in this endpoint. 
        
    """
    url = f"https://consumer-reports.p.rapidapi.com/product-groups/list"
    querystring = {}
    if size:
        querystring['size'] = size
    if productgrouptypeid:
        querystring['productGroupTypeId'] = productgrouptypeid
    if page:
        querystring['page'] = page
    if parentproductgroupid:
        querystring['parentProductGroupId'] = parentproductgroupid
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "consumer-reports.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def cars_get_images(modelyearid: int, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Get images of car model by year"
    modelyearid: The value of modelYearId field returned in .../cars/get-models endpoint
        
    """
    url = f"https://consumer-reports.p.rapidapi.com/cars/get-images"
    querystring = {'modelYearId': modelyearid, }
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "consumer-reports.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def cars_get_recalls(modelyearid: int, size: int=20, page: int=0, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Get recalls relating to a car model year"
    modelyearid: The value of modelYearId field returned in .../cars/get-models endpoint
        size: The number of items per response, for paging purpose
        page: The page index starting from 0, for paging purpose
        
    """
    url = f"https://consumer-reports.p.rapidapi.com/cars/get-recalls"
    querystring = {'modelYearId': modelyearid, }
    if size:
        querystring['size'] = size
    if page:
        querystring['page'] = page
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "consumer-reports.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def cars_detail(modelyearid: int, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Get detailed information of car model by year"
    modelyearid: The value of modelYearId field returned in .../cars/get-models endpoint
        
    """
    url = f"https://consumer-reports.p.rapidapi.com/cars/detail"
    querystring = {'modelYearId': modelyearid, }
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "consumer-reports.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def cars_get_models(modelid: int, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Get model generations"
    modelid: The value of carModels -> id field returned in .../search endpoint
        
    """
    url = f"https://consumer-reports.p.rapidapi.com/cars/get-models"
    querystring = {'modelId': modelid, }
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "consumer-reports.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def search(query: str, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Search for brand, car, product, etc... by term or phrase"
    query: Any term or phrase that you are familiar with
        
    """
    url = f"https://consumer-reports.p.rapidapi.com/search"
    querystring = {'query': query, }
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "consumer-reports.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

