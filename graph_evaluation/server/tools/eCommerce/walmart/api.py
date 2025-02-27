import requests
import json
from datetime import date, datetime, timedelta
import os

from typing import Optional, Dict, Union, List


def productdetails_deprecated(usitemid: int, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "(!Deprecated!) Product details. Use `productDescription` instead."
    usitemid: The value of usItemId field returned in `/search` endpoint.
        
    """
    url = f"https://walmart2.p.rapidapi.com/productDetails"
    querystring = {'usItemId': usitemid, }
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "walmart2.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def search_search_by_keyword(query: str, page: int=None, zip: int=None, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Search products by Keyword"
    query: Product name or brand to search
        page: Page number if at the previous response `totalPages` > 1
        
    """
    url = f"https://walmart2.p.rapidapi.com/search"
    querystring = {'query': query, }
    if page:
        querystring['page'] = page
    if zip:
        querystring['zip'] = zip
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "walmart2.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def productreviews_product_reviews(usitemid: int, sort: str=None, page: int=1, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Product reviews"
    usitemid: The value of usItemId field returned in `/search` endpoint.
        sort: Sort reviews by:
 `RELEVANT` - most relevant, *default*
 `HELPFUL` - most helpful
 `NEWEST_FIRST` - newest to oldest
 `OLDEST_FIRST` - oldest to newest
 `HIGH_RATE_FIRST` - high to low rating
 `LOW_RATE_FIRST` - low to high rating
        page: Pagination. 
20 reviews are limit per page.
        
    """
    url = f"https://walmart2.p.rapidapi.com/productReviews"
    querystring = {'usItemId': usitemid, }
    if sort:
        querystring['sort'] = sort
    if page:
        querystring['page'] = page
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "walmart2.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def productdescription_product_description(usitemid: int, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Get product description. Include ingredients and specifications."
    usitemid: The value of usItemId field returned in `/search` endpoint.
        
    """
    url = f"https://walmart2.p.rapidapi.com/productDescription"
    querystring = {'usItemId': usitemid, }
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "walmart2.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def searchv2_search_by_keyword_v2(query: str, page: int=None, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Search products by Keyword"
    query: Product name or brand to search
        page: Page number if at the previous response `totalPages` > 1
        
    """
    url = f"https://walmart2.p.rapidapi.com/searchV2"
    querystring = {'query': query, }
    if page:
        querystring['page'] = page
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "walmart2.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def searchbyupc_search_by_upc(upc: str, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Search product by UPC"
    upc: Product UPC
        
    """
    url = f"https://walmart2.p.rapidapi.com/searchByUPC"
    querystring = {'upc': upc, }
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "walmart2.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

