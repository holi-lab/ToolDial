import requests
import json
from datetime import date, datetime, timedelta
import os

from typing import Optional, Dict, Union, List


def get_all_products(limit: int=100, page: int=1, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "This will return all the products available.
		You can use products?limit=100&&page=1 for pagination"
    
    """
    url = f"https://e-commerce12.p.rapidapi.com/products"
    querystring = {}
    if limit:
        querystring['limit'] = limit
    if page:
        querystring['page'] = page
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "e-commerce12.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def get_product_by_id(is_id: str, similarproducts: bool=None, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "This will get the product with the specified ID, pagination is disabled on this page
		To get similar products just pass similarProducts=true in your query"
    
    """
    url = f"https://e-commerce12.p.rapidapi.com/products/getproductbyid/{is_id}"
    querystring = {}
    if similarproducts:
        querystring['similarProducts'] = similarproducts
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "e-commerce12.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def top_sales_by_category(category: str, sales: int=100, limit: int=100, page: int=1, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "This will get all products with over 100 sales in the specified category"
    
    """
    url = f"https://e-commerce12.p.rapidapi.com/categories/topsales"
    querystring = {'category': category, }
    if sales:
        querystring['sales'] = sales
    if limit:
        querystring['limit'] = limit
    if page:
        querystring['page'] = page
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "e-commerce12.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def top_rated_by_category(category: str, rating: int=4, limit: int=100, page: int=1, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "This will get the top rated items in the category specified in the query params
		The default is items that have over 4 ratings  if rating is not specified in the query params"
    
    """
    url = f"https://e-commerce12.p.rapidapi.com/categories/toprated"
    querystring = {'category': category, }
    if rating:
        querystring['rating'] = rating
    if limit:
        querystring['limit'] = limit
    if page:
        querystring['page'] = page
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "e-commerce12.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def get_all_categories(toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "This will get all available categories"
    
    """
    url = f"https://e-commerce12.p.rapidapi.com/categories/"
    querystring = {}
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "e-commerce12.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def top_sold_products(limit: int=100, page: int=1, sales: int=100, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "This will display all the top products depending on the number of sales in the query params, if its not available it will display all products with over 100 sales"
    
    """
    url = f"https://e-commerce12.p.rapidapi.com/products/topsales"
    querystring = {}
    if limit:
        querystring['limit'] = limit
    if page:
        querystring['page'] = page
    if sales:
        querystring['sales'] = sales
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "e-commerce12.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def get_top_rated_products(page: int=1, limit: int=50, rating: int=4, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "This will get all the products whose ratings are greater than or equal to the specified rating in the query params"
    
    """
    url = f"https://e-commerce12.p.rapidapi.com/products/toprated"
    querystring = {}
    if page:
        querystring['page'] = page
    if limit:
        querystring['limit'] = limit
    if rating:
        querystring['rating'] = rating
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "e-commerce12.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def get_products_by_category(category: str, page: int=1, limit: int=50, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "This will get all products under different categories"
    
    """
    url = f"https://e-commerce12.p.rapidapi.com/categories/getitemsbycategory"
    querystring = {'category': category, }
    if page:
        querystring['page'] = page
    if limit:
        querystring['limit'] = limit
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "e-commerce12.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def get_product_by_name(limit: int=50, name: str='red', page: int=1, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "This endpoint will return any product that contains the given name"
    
    """
    url = f"https://e-commerce12.p.rapidapi.com/products/getproductbyname/search"
    querystring = {}
    if limit:
        querystring['limit'] = limit
    if name:
        querystring['name'] = name
    if page:
        querystring['page'] = page
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "e-commerce12.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

