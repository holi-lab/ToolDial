import requests
import json
from datetime import date, datetime, timedelta
import os

from typing import Optional, Dict, Union, List


def get_asin_by_gtin(gtin: str, country: str, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Search product by GTIN - Global Trade Item Number (e.g. 0194252099537) and get an ASIN (Amazon Standard Identification Number). Use ASIN for further searches."
    gtin: Search GTIN
        country: The marketplace country
        
    """
    url = f"https://amazon-merchant-data.p.rapidapi.com/get-asin"
    querystring = {'gtin': gtin, 'country': country, }
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "amazon-merchant-data.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def search_offers_by_asin(country: str, asin: str, page: int=1, condition: str=None, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Search Amazon by ASIN and return offers with prices."
    country: The marketplace country
        asin: ASIN value
        page: Page number of results
        condition: Product condition
        
    """
    url = f"https://amazon-merchant-data.p.rapidapi.com/get-offers"
    querystring = {'country': country, 'asin': asin, }
    if page:
        querystring['page'] = page
    if condition:
        querystring['condition'] = condition
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "amazon-merchant-data.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def get_product_reviews(country: str, asin: str, filterbystar: str=None, mediatype: str=None, sortby: str=None, filterbylanguage: str=None, reviewertype: str=None, filterbykeyword: str=None, page: int=1, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Search reviews by ASIN and country"
    country: The marketplace country
        asin: Search ASIN
        filterbystar: Filter by rating
        mediatype: Filter by media type
        sortby: Sorting parameter
        filterbylanguage: Filter by language
        reviewertype: Filter by reviewer type
        filterbykeyword: Filter by keyword
        page: Page number of results
        
    """
    url = f"https://amazon-merchant-data.p.rapidapi.com/get-reviews"
    querystring = {'country': country, 'asin': asin, }
    if filterbystar:
        querystring['filterByStar'] = filterbystar
    if mediatype:
        querystring['mediaType'] = mediatype
    if sortby:
        querystring['sortBy'] = sortby
    if filterbylanguage:
        querystring['filterByLanguage'] = filterbylanguage
    if reviewertype:
        querystring['reviewerType'] = reviewertype
    if filterbykeyword:
        querystring['filterByKeyword'] = filterbykeyword
    if page:
        querystring['page'] = page
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "amazon-merchant-data.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def search_products_by_term(term: str, country: str, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Search by term (e.g. iphone 12) and get results"
    term: Search term
        country: The marketplace country
        
    """
    url = f"https://amazon-merchant-data.p.rapidapi.com/search-products"
    querystring = {'term': term, 'country': country, }
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "amazon-merchant-data.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def search_product_by_asin(asin: str, country: str, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Search product by a marketplace product id (Amazon ASIN, e.g. B09G98X7GV). A successful response returns product details."
    asin: Search ASIN
        country: The marketplace country
        
    """
    url = f"https://amazon-merchant-data.p.rapidapi.com/get-product"
    querystring = {'asin': asin, 'country': country, }
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "amazon-merchant-data.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

