import requests
import json
from datetime import date, datetime, timedelta
import os

from typing import Optional, Dict, Union, List


def categories(toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    " "
    
    """
    url = f"https://ali-express1.p.rapidapi.com/categories"
    querystring = {}
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "ali-express1.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def product_product_id(product_id: str, language: str='en', toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    " "
    
    """
    url = f"https://ali-express1.p.rapidapi.com/product/{product_id}"
    querystring = {}
    if language:
        querystring['language'] = language
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "ali-express1.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def product_product_id_feedback(product_id: str, country: str='us', page: str='1', type: str='all', toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    " "
    country: Country code. used for local reviews. Use type 'local'
        type: type of feedback
        
    """
    url = f"https://ali-express1.p.rapidapi.com/product/{product_id}/feedback/"
    querystring = {}
    if country:
        querystring['country'] = country
    if page:
        querystring['page'] = page
    if type:
        querystring['type'] = type
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "ali-express1.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def productsbycategoryv2_category_id(category_id: str, sort_type: str='default', page: int=1, page_size: int=20, sort_order: str='default', toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    " "
    
    """
    url = f"https://ali-express1.p.rapidapi.com/productsByCategoryV2/{category_id}"
    querystring = {}
    if sort_type:
        querystring['sort_type'] = sort_type
    if page:
        querystring['page'] = page
    if page_size:
        querystring['page_size'] = page_size
    if sort_order:
        querystring['sort_order'] = sort_order
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "ali-express1.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def search(query: str, page: str='1', toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    " "
    page: first item number in case of pagination
        
    """
    url = f"https://ali-express1.p.rapidapi.com/search"
    querystring = {'query': query, }
    if page:
        querystring['page'] = page
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "ali-express1.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def store_seller_id_feedback(seller_id: str, country: str='us', page: str='1', type: str='all', toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    " "
    seller_id: Can for example be obtained from a get_prodcut: 'storeModule' > 'sellerAdminSeq'
        country: Country code. used for local reviews. Use type 'local'
        type: type of feedback
        
    """
    url = f"https://ali-express1.p.rapidapi.com/store/{seller_id}/feedback/"
    querystring = {}
    if country:
        querystring['country'] = country
    if page:
        querystring['page'] = page
    if type:
        querystring['type'] = type
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "ali-express1.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def store_seller_id_information(seller_id: str, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    " "
    seller_id: Can for example be obtained from a get_prodcut: 'storeModule' > 'sellerAdminSeq'
        
    """
    url = f"https://ali-express1.p.rapidapi.com/store/{seller_id}/information"
    querystring = {}
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "ali-express1.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def store_get_categories_store_id(store_id: str, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    " "
    store_id: Can for example be obtained from get_store_information: 'data' > 'shopInfo' > 'shopId'
        
    """
    url = f"https://ali-express1.p.rapidapi.com/store/get_categories/{store_id}"
    querystring = {}
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "ali-express1.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def store_get_products_by_categories(store_id: str, group_id: str, seller_id: str, page: str='1', toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    " "
    store_id: Can for example be obtained from get_store_information: 'data' > 'shopInfo' > 'shopId'
        group_id: Can for example be obtained from get_store_information: 'data' > 'shopInfo' > 'shopId'
        seller_id: Can for example be obtained from a get_prodcut: 'storeModule' > 'sellerAdminSeq'
        
    """
    url = f"https://ali-express1.p.rapidapi.com/store/get_products_by_categories/"
    querystring = {'store_id': store_id, 'group_id': group_id, 'seller_id': seller_id, }
    if page:
        querystring['page'] = page
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "ali-express1.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def shipping_product_id(destination_country: str, product_id: str, min_price: str='0.00', count: str='1', toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Returns shipping information"
    min_price: Product price. 

If not provided, it can happen that extra shipping methods are returned which are not actually offered
        
    """
    url = f"https://ali-express1.p.rapidapi.com/shipping/{product_id}"
    querystring = {'destination_country': destination_country, }
    if min_price:
        querystring['min_price'] = min_price
    if count:
        querystring['count'] = count
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "ali-express1.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

