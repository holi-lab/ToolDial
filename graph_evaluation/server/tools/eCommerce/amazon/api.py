import requests
import json
from datetime import date, datetime, timedelta
import os

from typing import Optional, Dict, Union, List


def single_product_details(asin: str, country: str='US', toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Get product details and pricing information by ASIN."
    
    """
    url = f"https://amazon23.p.rapidapi.com/product-details"
    querystring = {'asin': asin, }
    if country:
        querystring['country'] = country
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "amazon23.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def product_search(country: str='US', page: int=None, query: str='xbox', toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Search products by query. Around 20 items returned per page."
    country: Amazon country. Allowed values: 'US', 'AU', 'BR', 'CA', 'CN', 'FR', 'DE',
        'IN', 'IT', 'MX', 'NL', 'SG', 'ES', 'TR', 'AE', 'GB', 'JP'
        page: Paginate over results by passing 2, 3, 4, etc for page number.
        
    """
    url = f"https://amazon23.p.rapidapi.com/product-search"
    querystring = {}
    if country:
        querystring['country'] = country
    if page:
        querystring['page'] = page
    if query:
        querystring['query'] = query
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "amazon23.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def single_product_reviews(asin: str, sort_by: str, country: str='US', page: int=None, rating: str=None, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Get product reviews by ASIN"
    country: Amazon country. Allowed values: 'US', 'AU', 'BR', 'CA', 'CN', 'FR', 'DE',
        'IN', 'IT', 'MX', 'NL', 'SG', 'ES', 'TR', 'AE', 'GB', 'JP'
        page: Paginate over results by passing 2,3,4, etc page numbers
        rating: Rating min and max, dash separated. Example: \"1-3\" to get only reviews with 1,2,3 ratings or \"1-1\" - to get only reviews with 1 star rating.
        
    """
    url = f"https://amazon23.p.rapidapi.com/reviews"
    querystring = {'asin': asin, 'sort_by': sort_by, }
    if country:
        querystring['country'] = country
    if page:
        querystring['page'] = page
    if rating:
        querystring['rating'] = rating
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "amazon23.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

