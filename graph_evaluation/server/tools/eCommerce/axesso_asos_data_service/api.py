import requests
import json
from datetime import date, datetime, timedelta
import os

from typing import Optional, Dict, Union, List


def product_reviews(page: int, domaincode: str, productid: str, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Fetch reviews for productId"
    
    """
    url = f"https://axesso-asos-data-service.p.rapidapi.com/aso/lookup-product-reviews"
    querystring = {'page': page, 'domainCode': domaincode, 'productId': productid, }
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "axesso-asos-data-service.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def search_products(domaincode: str, page: int, keyword: str, sortby: str='discountPercentage-desc', toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Search products for provided keyword"
    domaincode: Store / country. Valid values are `us` and `de`.
        sortby: Valid values are \"freshness\", \"price-desc\" or \"price-asc\"
        
    """
    url = f"https://axesso-asos-data-service.p.rapidapi.com/aso/search-by-keyword"
    querystring = {'domainCode': domaincode, 'page': page, 'keyword': keyword, }
    if sortby:
        querystring['sortBy'] = sortby
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "axesso-asos-data-service.p.rapidapi.com"
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
    url = f"https://axesso-asos-data-service.p.rapidapi.com/aso/lookup-product-details"
    querystring = {'url': url, }
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "axesso-asos-data-service.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

