import requests
import json
from datetime import date, datetime, timedelta
import os

from typing import Optional, Dict, Union, List


def products_v3_list_similarities(is_id: int, sizeschema: str='US', currency: str='USD', lang: str='en-US', store: str='US', toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "List similar products by product id"
    id: The value of id field that returned in .../products/v2/list
        sizeschema: Get suitable value from .../countries/list endpoint
        currency: Get suitable value from .../countries/list endpoint
        lang: The language code
        store: The store code gotten from .../countries/list
        
    """
    url = f"https://asos2.p.rapidapi.com/products/v3/list-similarities"
    querystring = {'id': is_id, }
    if sizeschema:
        querystring['sizeSchema'] = sizeschema
    if currency:
        querystring['currency'] = currency
    if lang:
        querystring['lang'] = lang
    if store:
        querystring['store'] = store
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "asos2.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def products_v3_detail(is_id: int, currency: str='USD', sizeschema: str='US', store: str='US', lang: str='en-US', toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Get detail information of product by id"
    id: Get id value from products/list API
        currency: Get suitable value from countries/list API
        sizeschema: Get suitable value from countries/list API
        store: Get suitable \"store\" value from countries/list API, this param is not sideId as in products/list API
        lang: Get suitable value from countries/list API
        
    """
    url = f"https://asos2.p.rapidapi.com/products/v3/detail"
    querystring = {'id': is_id, }
    if currency:
        querystring['currency'] = currency
    if sizeschema:
        querystring['sizeSchema'] = sizeschema
    if store:
        querystring['store'] = store
    if lang:
        querystring['lang'] = lang
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "asos2.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def countries_list(lang: str='en-US', toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "List countries that Asos supports selling products"
    lang: The language code
        
    """
    url = f"https://asos2.p.rapidapi.com/countries/list"
    querystring = {}
    if lang:
        querystring['lang'] = lang
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "asos2.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def products_list_deprecated(limit: str, store: int, categoryid: str, offset: str, attribute_1046: int=None, min: int=None, base_colour: int=None, brand: str=None, max: int=None, sort: str='freshness', attribute_10155: str=None, lang: str='en-US', country: str='US', size: int=None, attribute_1047: str=None, currency: str='USD', sizeschema: str='US', toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "List products, search products with options and filters"
    limit: The number of items per page
        store: Get value from siteId field from countries/list API
        categoryid: Get value from categories/list API
        offset: The offset to skip already viewed products
        attribute_1046: Filter option by Style, you can pass this params multiple times, such as : ...&attribute_1046= 8391&attribute_1046=8392
        min: Filter option by Min Price ( integer value )
        base_colour: Filter option by Color, look for "id" of "facetValues" in the response of this API. You can pass this params multiple times, such as : ...&base_colour= 8391&base_colour=8392
        brand: Filter option by Brand, look for "id" of "facetValues" in the response of this API. You can pass this params multiple times, such as : ...&brand= 8391&brand=8392
        max: Filter option by Max Price ( integer value )
        sort: One of the following pricedesc|priceasc|freshness (Newest)
        attribute_10155: Filter option by Range, look for "id" of "facetValues" in the response of this API. You can pass this params multiple times, such as : ...&attribute_10155= 8391&attribute_10155=8392
        lang: The language code
        country: The country code
        size: Filter option by Size, look for "id" of "facetValues" in the response of this API. You can pass this params multiple times, such as : ...&size= 8391&size=8392
        attribute_1047: Filter option by Product Type, you can pass this params multiple times, such as : ...&attribute_1047= 8391&attribute_1047=8392
        currency: Get suitable value from countries/list API
        sizeschema: Get suitable value from countries/list API
        
    """
    url = f"https://asos2.p.rapidapi.com/products/list"
    querystring = {'limit': limit, 'store': store, 'categoryId': categoryid, 'offset': offset, }
    if attribute_1046:
        querystring['attribute_1046'] = attribute_1046
    if min:
        querystring['min'] = min
    if base_colour:
        querystring['base_colour'] = base_colour
    if brand:
        querystring['brand'] = brand
    if max:
        querystring['max'] = max
    if sort:
        querystring['sort'] = sort
    if attribute_10155:
        querystring['attribute_10155'] = attribute_10155
    if lang:
        querystring['lang'] = lang
    if country:
        querystring['country'] = country
    if size:
        querystring['size'] = size
    if attribute_1047:
        querystring['attribute_1047'] = attribute_1047
    if currency:
        querystring['currency'] = currency
    if sizeschema:
        querystring['sizeSchema'] = sizeschema
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "asos2.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def products_v2_list(store: str, offset: int, categoryid: int, limit: int, attribute_1046: str=None, pricemin: int=None, country: str='US', attribute_10147: str=None, sort: str='freshness', q: str=None, base_colour: str=None, range: str=None, attribute_1047: str=None, currency: str='USD', attribute_10155: str=None, pricemax: int=None, sizeschema: str='US', brand: str=None, size: str=None, lang: str='en-US', toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "List products, search products with options and filters version 2"
    store: Get value from store field from countries/list API
        offset: The offset to skip already viewed products
        categoryid: The value of categoryId field from categories/list API
        limit: The number of items per page
        attribute_1046: Filter option by Style (separated by comma for multiple values), such as : ...&attribute_1046=8391,8392
        pricemin: Filter option by Min Price ( integer value )
        country: The country code
        attribute_10147: Filter option by Leather / Non Leather (separated by comma for multiple values), such as : ...&attribute_10147=8391,8392
        sort: One of the following pricedesc|priceasc|freshness (Newest)
        q: Search for products by name (do not use this parameter with categoryId)
        base_colour: Filter option by Color, look for "id" of "facetValues" in the response of this API (separated by comma for multiple values), such as : ...&base_colour=8391,8392
        range: Filter option by Sale/New Season (separated by comma for multiple values), such as : ...&range=new_season
        attribute_1047: Filter option by Product Type (separated by comma for multiple values), such as : ...&attribute_1047=8391,8392
        currency: Get suitable value from countries/list API
        attribute_10155: Filter option by Range (separated by comma for multiple values), such as : ...&attribute_10155=8391,8392
        pricemax: Filter option by Max Price ( integer value )
        sizeschema: Get suitable value from countries/list API
        brand: Filter option by Brand, look for "id" of "facetValues" in the response of this API (separated by comma for multiple values), such as : ...&brand=8391,8392
        size: Filter option by Size, look for "id" of "facetValues" in the response of this API (separated by comma for multiple values), such as : ...&size=8391,8392
        lang: The language code
        
    """
    url = f"https://asos2.p.rapidapi.com/products/v2/list"
    querystring = {'store': store, 'offset': offset, 'categoryId': categoryid, 'limit': limit, }
    if attribute_1046:
        querystring['attribute_1046'] = attribute_1046
    if pricemin:
        querystring['priceMin'] = pricemin
    if country:
        querystring['country'] = country
    if attribute_10147:
        querystring['attribute_10147'] = attribute_10147
    if sort:
        querystring['sort'] = sort
    if q:
        querystring['q'] = q
    if base_colour:
        querystring['base_colour'] = base_colour
    if range:
        querystring['range'] = range
    if attribute_1047:
        querystring['attribute_1047'] = attribute_1047
    if currency:
        querystring['currency'] = currency
    if attribute_10155:
        querystring['attribute_10155'] = attribute_10155
    if pricemax:
        querystring['priceMax'] = pricemax
    if sizeschema:
        querystring['sizeSchema'] = sizeschema
    if brand:
        querystring['brand'] = brand
    if size:
        querystring['size'] = size
    if lang:
        querystring['lang'] = lang
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "asos2.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def products_detail_deprecated(is_id: int, lang: str='en-US', store: str='US', currency: int=None, sizeschema: str='US', toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Get detail information of product by id"
    id: Get id value from products/list API
        lang: Get suitable value from countries/list API
        store: Get suitable "store" value from countries/list API, this param is not sideId as in products/list API
        currency: Get suitable value from countries/list API
        sizeschema: Get suitable value from countries/list API
        
    """
    url = f"https://asos2.p.rapidapi.com/products/detail"
    querystring = {'id': is_id, }
    if lang:
        querystring['lang'] = lang
    if store:
        querystring['store'] = store
    if currency:
        querystring['currency'] = currency
    if sizeschema:
        querystring['sizeSchema'] = sizeschema
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "asos2.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def products_v2_list_similarities_deprecating(is_id: int, store: str='US', toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "List similar products by product id"
    id: The value of id field that returned in .../products/v2/list
        store: The store code gotten from .../countries/list
        
    """
    url = f"https://asos2.p.rapidapi.com/products/v2/list-similarities"
    querystring = {'id': is_id, }
    if store:
        querystring['store'] = store
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "asos2.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def v2_auto_complete(q: str, store: str='US', country: str='US', currency: str='USD', sizeschema: str='US', lang: str='en-US', toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Get auto complete suggestion by product name"
    q: Name of products
        store: The store code gotten from .../countries/list API
        country: The country code gotten from .../countries/list API
        currency: The currency code gotten from .../countries/list API
        sizeschema: The sizeSchema code gotten from .../countries/list API
        lang: The language code gotten from .../countries/list API
        
    """
    url = f"https://asos2.p.rapidapi.com/v2/auto-complete"
    querystring = {'q': q, }
    if store:
        querystring['store'] = store
    if country:
        querystring['country'] = country
    if currency:
        querystring['currency'] = currency
    if sizeschema:
        querystring['sizeSchema'] = sizeschema
    if lang:
        querystring['lang'] = lang
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "asos2.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def categories_list(lang: str='en-US', country: str='US', toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "List categories from Asos"
    lang: The language code
        country: The two letters country code
        
    """
    url = f"https://asos2.p.rapidapi.com/categories/list"
    querystring = {}
    if lang:
        querystring['lang'] = lang
    if country:
        querystring['country'] = country
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "asos2.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

