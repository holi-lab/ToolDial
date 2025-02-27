import requests
import json
from datetime import date, datetime, timedelta
import os

from typing import Optional, Dict, Union, List


def get_all_the_records(page: str='2', toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "This endpoint returns all the records of a specific page... This will return simple returns fast.
		
		**Parameters:**
		-> page"
    
    """
    url = f"https://shopify-stores.p.rapidapi.com/getAllRecord"
    querystring = {}
    if page:
        querystring['page'] = page
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "shopify-stores.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def search_stores(page: str=None, search: str=None, part: str=None, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "This endpoint will search for stores on Shopify Database.
		
		**Parameters:**
		-> page
		-> part (for each page has 5 parts). i.e: page=2 has part=1, or 2 to 5. In case you want all the results of a particular page at once, use part=0. This will return all the records of the page, but it will take 1-2 minutes to finish.
		-> search (The keyword to search)"
    
    """
    url = f"https://shopify-stores.p.rapidapi.com/searchStores"
    querystring = {}
    if page:
        querystring['page'] = page
    if search:
        querystring['search'] = search
    if part:
        querystring['part'] = part
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "shopify-stores.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def get_all_stores(page: str='2', part: str='2', toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "This endpoint returns all the Shopify stores...
		
		**Parameters:**
		->  page (the number of page to return)
		-> part (for each page has 5 parts, which means. i.e: page=1 has part=1, or 2 to 5) to retrieve data fast and easy pagination...
		 
		In case you want all the results of a particular page at once... Use part=0, this will return all the records of a page around 50, but will take 1-2 min to finish."
    
    """
    url = f"https://shopify-stores.p.rapidapi.com/getStores"
    querystring = {}
    if page:
        querystring['page'] = page
    if part:
        querystring['part'] = part
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "shopify-stores.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

