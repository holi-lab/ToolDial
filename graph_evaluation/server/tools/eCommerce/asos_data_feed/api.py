import requests
import json
from datetime import date, datetime, timedelta
import os

from typing import Optional, Dict, Union, List


def search(country: str, q: str, pricemax: str=None, sizeschema: str=None, filter: str=None, pricemin: str=None, page: int=None, currency: str=None, brand: str=None, lang: str=None, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Search for products with filters"
    country: `US`, `GB`, `IN`, etc.
Other available country code you can get from `/countries` endpoint.
        q: Search for items and brands
        sizeschema: Data from `/countries` endpoint.
        filter: Filter option.
*Format: filter[range]=new_season,sale&filter[size]=98*
The filters are unique and generated for you `q` parameter, then  returned right in the response.

        page: Page number of search results.
        currency: Data from `/countries` endpoint.
        lang: Data from `/countries` endpoint.
        
    """
    url = f"https://asos-data-feed.p.rapidapi.com/search"
    querystring = {'country': country, 'q': q, }
    if pricemax:
        querystring['priceMax'] = pricemax
    if sizeschema:
        querystring['sizeSchema'] = sizeschema
    if filter:
        querystring['filter'] = filter
    if pricemin:
        querystring['priceMin'] = pricemin
    if page:
        querystring['page'] = page
    if currency:
        querystring['currency'] = currency
    if brand:
        querystring['brand'] = brand
    if lang:
        querystring['lang'] = lang
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "asos-data-feed.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def product_reviews(country: str, is_id: int, offset: int=None, lang: str=None, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Product reviews."
    country: `US`, `GB`, `IN`, etc.
Other available country code you can get from `/countries` endpoint.
        
    """
    url = f"https://asos-data-feed.p.rapidapi.com/product/reviews"
    querystring = {'country': country, 'id': is_id, }
    if offset:
        querystring['offset'] = offset
    if lang:
        querystring['lang'] = lang
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "asos-data-feed.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def product_details(country: str, is_id: int, lang: str=None, currency: str=None, sizeschema: str=None, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Get information about the product by id."
    country: `US`, `GB`, `IN`, etc.
Other available country code you can get from `/countries` endpoint.
        
    """
    url = f"https://asos-data-feed.p.rapidapi.com/product/details"
    querystring = {'country': country, 'id': is_id, }
    if lang:
        querystring['lang'] = lang
    if currency:
        querystring['currency'] = currency
    if sizeschema:
        querystring['sizeSchema'] = sizeschema
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "asos-data-feed.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def countries(toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Countries that Asos supports selling products"
    
    """
    url = f"https://asos-data-feed.p.rapidapi.com/countries"
    querystring = {}
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "asos-data-feed.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

