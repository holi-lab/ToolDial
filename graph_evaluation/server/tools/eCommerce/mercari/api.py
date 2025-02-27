import requests
import json
from datetime import date, datetime, timedelta
import os

from typing import Optional, Dict, Union, List


def get_item_information(item_id: str, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Get an item information by item id.
		item id starts with a letter "m" which is show on URL.
		for example:
		https://www.mercari.com/us/item/m87613775748
		
		m87613775748 is the item id you are looking for."
    
    """
    url = f"https://mercari.p.rapidapi.com/Mercari/GetItem"
    querystring = {'item_id': item_id, }
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "mercari.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def get_seller_information(seller_id: str, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Get a seller information by seller id.
		Seller id is a bunch of number which is show on URL.
		for example:
		https://www.mercari.com/u/589609090
		
		589609090 is the seller id you are looking for."
    
    """
    url = f"https://mercari.p.rapidapi.com/Mercari/GetSeller"
    querystring = {'seller_id': seller_id, }
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "mercari.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def get_parent_category(category_id: int, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "get a parent category of a sub category  by id"
    
    """
    url = f"https://mercari.p.rapidapi.com/Mercari/GetParentCategory"
    querystring = {'category_id': category_id, }
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "mercari.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def get_sub_categories(category_id: int, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "get all sub categories by parent category id"
    
    """
    url = f"https://mercari.p.rapidapi.com/Mercari/GetSubCategories"
    querystring = {'category_id': category_id, }
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "mercari.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def get_all_categories(toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Getting all parents and sub categories"
    
    """
    url = f"https://mercari.p.rapidapi.com/Mercari/GetAllCategories"
    querystring = {}
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "mercari.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def search_item(page: str, query: str, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Searching items keyword."
    
    """
    url = f"https://mercari.p.rapidapi.com/Mercari/Search"
    querystring = {'page': page, 'query': query, }
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "mercari.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def get_seller_s_items(page: str, seller_id: str, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Get seller's items by seller id"
    
    """
    url = f"https://mercari.p.rapidapi.com/Mercari/GetSellerItems"
    querystring = {'page': page, 'seller_id': seller_id, }
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "mercari.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def get_urls_by_search(query: str, page: str='1', toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "get the urls of all the items by searching the specific keyword."
    
    """
    url = f"https://mercari.p.rapidapi.com/Mercari/GetUrlsBySearch"
    querystring = {'query': query, }
    if page:
        querystring['page'] = page
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "mercari.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

