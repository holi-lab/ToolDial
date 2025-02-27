import requests
import json
from datetime import date, datetime, timedelta
import os

from typing import Optional, Dict, Union, List


def lookupsellerprices(page: str, domaincode: str, asin: str, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Lookup offer from different seller for the specified product"
    page: Number of pages to return
        domaincode: e.g. com / de / fr / co.uk
        
    """
    url = f"https://axesso-axesso-amazon-data-service-v1.p.rapidapi.com/v2/amz/amazon-lookup-prices"
    querystring = {'page': page, 'domainCode': domaincode, 'asin': asin, }
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "axesso-axesso-amazon-data-service-v1.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def searchbykeywordasin(keyword: str, page: str, domaincode: str, category: str=None, withcache: bool=None, excludesponsored: str='false', sortby: str='relevanceblender', toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Search products by keyword, returning all asins for the specified page."
    page: Parameter for Pagination
        domaincode: e.g. com / de / fr / co.uk
        category: Pass the amazon category which should used for the search. Valid category list can found on the amazon website on the search selection box. **Important: If the passed category is not a valid amazon category, the response will be empty**
        withcache: If true, proxy for same searchterm will be cached and used for next page. Might increase the response quality.
        excludesponsored: Set true to exclude sponsored products
        
    """
    url = f"https://axesso-axesso-amazon-data-service-v1.p.rapidapi.com/amz/amazon-search-by-keyword-asin"
    querystring = {'keyword': keyword, 'page': page, 'domainCode': domaincode, }
    if category:
        querystring['category'] = category
    if withcache:
        querystring['withCache'] = withcache
    if excludesponsored:
        querystring['excludeSponsored'] = excludesponsored
    if sortby:
        querystring['sortBy'] = sortby
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "axesso-axesso-amazon-data-service-v1.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def lookupreviews(asin: str, domaincode: str, page: int, sortby: str='recent', filters: str='reviewerType=avp_only_reviews;filterByStar=five_star', toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Lookup reviews from a product"
    filters: Semicolon seperated list of filters
        
    """
    url = f"https://axesso-axesso-amazon-data-service-v1.p.rapidapi.com/amz/amazon-lookup-reviews"
    querystring = {'asin': asin, 'domainCode': domaincode, 'page': page, }
    if sortby:
        querystring['sortBy'] = sortby
    if filters:
        querystring['filters'] = filters
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "axesso-axesso-amazon-data-service-v1.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def lookupseller(sellerid: str='AD97MR4NOW5CD', domaincode: str='com', toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Lookup seller details based on sellerid and domain"
    
    """
    url = f"https://axesso-axesso-amazon-data-service-v1.p.rapidapi.com/amz/amazon-lookup-seller"
    querystring = {}
    if sellerid:
        querystring['sellerId'] = sellerid
    if domaincode:
        querystring['domainCode'] = domaincode
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "axesso-axesso-amazon-data-service-v1.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def lookupproduct(url: str, merchant: str=None, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "request product information"
    merchant: merchant id
        
    """
    url = f"https://axesso-axesso-amazon-data-service-v1.p.rapidapi.com/amz/amazon-lookup-product"
    querystring = {'url': url, }
    if merchant:
        querystring['merchant'] = merchant
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "axesso-axesso-amazon-data-service-v1.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def lookupsellerproducts(page: int, domaincode: str, sellerid: str, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Lookup all products for a given seller page by page."
    
    """
    url = f"https://axesso-axesso-amazon-data-service-v1.p.rapidapi.com/amz/amazon-seller-products"
    querystring = {'page': page, 'domainCode': domaincode, 'sellerId': sellerid, }
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "axesso-axesso-amazon-data-service-v1.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def lookupbestseller(page: str='1', url: str='https://www.amazon.com/gp/movers-and-shakers/toys-and-games', toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Lookup best seller list by providing url and page."
    
    """
    url = f"https://axesso-axesso-amazon-data-service-v1.p.rapidapi.com/amz/amazon-best-sellers-list"
    querystring = {}
    if page:
        querystring['page'] = page
    if url:
        querystring['url'] = url
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "axesso-axesso-amazon-data-service-v1.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

