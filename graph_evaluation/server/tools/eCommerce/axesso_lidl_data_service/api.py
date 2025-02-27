import requests
import json
from datetime import date, datetime, timedelta
import os

from typing import Optional, Dict, Union, List


def product_rewiews(productid: str, page: int, sortby: str=None, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Fetch reviews for productId"
    sortby: Valid values are `most_relevant-desc`, `upvotes-desc`, `stars-asc`, `date-desc` and `date-asc`
        
    """
    url = f"https://axesso-lidl-data-service.p.rapidapi.com/ldl/lookup-product-reviews"
    querystring = {'productId': productid, 'page': page, }
    if sortby:
        querystring['sortBy'] = sortby
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "axesso-lidl-data-service.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def search_products(keyword: str, page: int, sortby: str=None, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Search products for provided keyword"
    sortby: Valid values are `price`, `price-desc`, `deliveryStartDate-desc`, `ratingScore-desc`, `sh_carts-desc` and  `discountPercentage-desc`
        
    """
    url = f"https://axesso-lidl-data-service.p.rapidapi.com/ldl/search-by-keyword"
    querystring = {'keyword': keyword, 'page': page, }
    if sortby:
        querystring['sortBy'] = sortby
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "axesso-lidl-data-service.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def product_details(url: str, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Fetch product details information by URL"
    
    """
    url = f"https://axesso-lidl-data-service.p.rapidapi.com/ldl/lookup-product-details"
    querystring = {'url': url, }
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "axesso-lidl-data-service.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

