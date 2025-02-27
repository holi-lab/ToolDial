import requests
import json
from datetime import date, datetime, timedelta
import os

from typing import Optional, Dict, Union, List


def reviews_list(size: str=None, is_picture: int=None, page: int=1, limit: int=20, comment_rank: int=None, color_id: str=None, sort: str='default', cat_id: str='1980', goods_spu: str='m22022854841', currency: str='USD', goods_id: str=None, language: str='en', country: str='US', toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "List reviews related to a product"
    size: One of the following : S|M|L|XL
        is_picture: Reviews must contain pictures, 0 or 1
        page: The page index, for paging purpose
        limit: The number of items per response, for paging purpose
        comment_rank: Filter comments by rank, from 1 to 5
        sort: One of the following : default|time_desc|time_asc
        cat_id: The value of cat&#95;id returned in .../products/list or .../products/search
        goods_spu: The value of 'productRelationID' returned in .../products/list or .../products/search
        currency: The 3-letter currency code
        goods_id: The value of 'goods_id' field returned in .../products/list or .../products/search endpoint
        language: The 2-letter language code
        country: The 2-letter country code
        
    """
    url = f"https://unofficial-shein.p.rapidapi.com/reviews/list"
    querystring = {}
    if size:
        querystring['size'] = size
    if is_picture:
        querystring['is_picture'] = is_picture
    if page:
        querystring['page'] = page
    if limit:
        querystring['limit'] = limit
    if comment_rank:
        querystring['comment_rank'] = comment_rank
    if color_id:
        querystring['color_id'] = color_id
    if sort:
        querystring['sort'] = sort
    if cat_id:
        querystring['cat_id'] = cat_id
    if goods_spu:
        querystring['goods_spu'] = goods_spu
    if currency:
        querystring['currency'] = currency
    if goods_id:
        querystring['goods_id'] = goods_id
    if language:
        querystring['language'] = language
    if country:
        querystring['country'] = country
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "unofficial-shein.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def products_get_extra_info(goods_id: str, brandcode: str='10001', cateid: str='1727', country_id: str='233', seriesid: str=None, brandbadge: str='SHEIN', language: str='en', sku: str='swdress07210415662', currency: str='USD', country: str='US', toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Get extra information of a product"
    goods_id: The value of 'goods_id' field returned in .../products/list or .../products/search endpoint
        brandcode: The value of brand&#95;code returned in .../products/list or .../products/search
        cateid: The value of cat&#95;id returned in .../products/list or .../products/search
        country_id: The value of 'id' field returned in .../countries/list
        seriesid: The value of 'seriesId' returned in .../products/list or .../products/search
        brandbadge: The value of brand&#95;badge returned in .../products/list or .../products/search
        language: The 2-letter language code
        sku: The value of 'goods&#95;sn' returned in .../products/list or .../products/search
        currency: The 3-letter currency code
        country: The 2-letter country code
        
    """
    url = f"https://unofficial-shein.p.rapidapi.com/products/get-extra-info"
    querystring = {'goods_id': goods_id, }
    if brandcode:
        querystring['brandCode'] = brandcode
    if cateid:
        querystring['cateId'] = cateid
    if country_id:
        querystring['country_id'] = country_id
    if seriesid:
        querystring['seriesId'] = seriesid
    if brandbadge:
        querystring['brandBadge'] = brandbadge
    if language:
        querystring['language'] = language
    if sku:
        querystring['sku'] = sku
    if currency:
        querystring['currency'] = currency
    if country:
        querystring['country'] = country
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "unofficial-shein.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def products_get_filters(min_price: int=None, filter: str=None, max_price: int=None, keywords: str=None, currency: str='USD', cat_id: str=None, country: str='US', language: str='en', toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "You use this endpoint to build up the filters dynamically"
    min_price: Filter by price
        filter: The value of 'attr&#95;filter' field OR {attr&#95;id}&#95;{attr&#95;value&#95;id} returned in .../products/get-filters endpoint. Separated by comma for multiple options. Ex : 87_1357-87_710,87_1352,etc...
        max_price: Filter by price
        keywords: You should use the value of 'cateName' or 'word' field returned in .../auto-complete endpoint for best results
        currency: The 3-letter currency code
        cat_id: The value of 'hrefTarget' returned in .../navigations/get-node-content endpoint
        country: The 2-letter country code
        language: The 2-letter language code
        
    """
    url = f"https://unofficial-shein.p.rapidapi.com/products/get-filters"
    querystring = {}
    if min_price:
        querystring['min_price'] = min_price
    if filter:
        querystring['filter'] = filter
    if max_price:
        querystring['max_price'] = max_price
    if keywords:
        querystring['keywords'] = keywords
    if currency:
        querystring['currency'] = currency
    if cat_id:
        querystring['cat_id'] = cat_id
    if country:
        querystring['country'] = country
    if language:
        querystring['language'] = language
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "unofficial-shein.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def products_search(keywords: str, limit: int=20, page: int=1, max_price: int=None, min_price: int=None, filter: str=None, sort: int=7, language: str='en', cat_id: str=None, country: str='US', currency: str='USD', toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Search products by keywords with options and filters"
    keywords: You should use the value of 'cateName' or 'word' field returned in .../auto-complete endpoint for best results
        limit: The number of items per response, for paging purpose
        page: The page index, for paging purpose
        max_price: Filter by price
        min_price: Filter by price
        filter: The value of 'attr&#95;filter' field OR {attr&#95;id}&#95;{attr&#95;value&#95;id} returned in .../products/get-filters endpoint. Separated by comma for multiple options. Ex : 87_1357-87_710,87_1352,etc...
        sort: One of the following : 0-Recommend|7-Top rated|8-Most popular|9-New arrivals|10-Price low to high|11-Price high to low
        language: The 2-letter language code
        cat_id: The value of 'hrefTarget' returned in .../navigations/get-node-content endpoint
        country: The 2-letter country code
        currency: The 3-letter currency code
        
    """
    url = f"https://unofficial-shein.p.rapidapi.com/products/search"
    querystring = {'keywords': keywords, }
    if limit:
        querystring['limit'] = limit
    if page:
        querystring['page'] = page
    if max_price:
        querystring['max_price'] = max_price
    if min_price:
        querystring['min_price'] = min_price
    if filter:
        querystring['filter'] = filter
    if sort:
        querystring['sort'] = sort
    if language:
        querystring['language'] = language
    if cat_id:
        querystring['cat_id'] = cat_id
    if country:
        querystring['country'] = country
    if currency:
        querystring['currency'] = currency
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "unofficial-shein.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def products_get_reviews(goods_spu: str='m22022854841', cat_id: str='1727', sku: str='rm2202285484176751', currency: str='USD', goods_id: str='10196865', language: str='en', country: str='US', toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Get brief reviews of a product"
    goods_spu: The value of 'productRelationID' returned in .../products/list or .../products/search
        cat_id: The value of cat&#95;id returned in .../products/list or .../products/search
        sku: The value of 'goods&#95;sn' returned in .../products/list or .../products/search
        currency: The 3-letter currency code
        goods_id: The value of 'goods_id' field returned in .../products/list or .../products/search endpoint
        language: The 2-letter language code
        country: The 2-letter country code
        
    """
    url = f"https://unofficial-shein.p.rapidapi.com/products/get-reviews"
    querystring = {}
    if goods_spu:
        querystring['goods_spu'] = goods_spu
    if cat_id:
        querystring['cat_id'] = cat_id
    if sku:
        querystring['sku'] = sku
    if currency:
        querystring['currency'] = currency
    if goods_id:
        querystring['goods_id'] = goods_id
    if language:
        querystring['language'] = language
    if country:
        querystring['country'] = country
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "unofficial-shein.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def products_detail(goods_id: str, currency: str='USD', country: str='US', language: str='en', toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Get detailed information of a product"
    goods_id: The value of 'goods_id' field returned in .../products/list or .../products/search endpoint
        currency: The 3-letter currency code
        country: The 2-letter country code
        language: The 2-letter language code
        
    """
    url = f"https://unofficial-shein.p.rapidapi.com/products/detail"
    querystring = {'goods_id': goods_id, }
    if currency:
        querystring['currency'] = currency
    if country:
        querystring['country'] = country
    if language:
        querystring['language'] = language
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "unofficial-shein.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def products_list(adp: str, cat_id: str, max_price: int=None, sort: int=7, min_price: int=None, filter: str=None, currency: str='USD', page: int=1, limit: int=20, country: str='US', language: str='en', toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "List products by a category with options and filters"
    adp: The value of 'goodsId' returned in .../navigations/get-node-content endpoint
        cat_id: The value of 'hrefTarget' returned in .../navigations/get-node-content endpoint
        max_price: Filter by price
        sort: One of the following : 0-Recommend|7-Top rated|8-Most popular|9-New arrivals|10-Price low to high|11-Price high to low
        min_price: Filter by price
        filter: The value of 'attr&#95;filter' field OR {attr&#95;id}&#95;{attr&#95;value&#95;id} returned in .../products/get-filters endpoint. Separated by comma for multiple options. Ex : 87_1357-87_710,87_1352,etc...
        currency: The 3-letter currency code
        page: The page index, for paging purpose
        limit: The number of items per response, for paging purpose
        country: The 2-letter country code
        language: The 2-letter language code
        
    """
    url = f"https://unofficial-shein.p.rapidapi.com/products/list"
    querystring = {'adp': adp, 'cat_id': cat_id, }
    if max_price:
        querystring['max_price'] = max_price
    if sort:
        querystring['sort'] = sort
    if min_price:
        querystring['min_price'] = min_price
    if filter:
        querystring['filter'] = filter
    if currency:
        querystring['currency'] = currency
    if page:
        querystring['page'] = page
    if limit:
        querystring['limit'] = limit
    if country:
        querystring['country'] = country
    if language:
        querystring['language'] = language
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "unofficial-shein.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def navigations_get_node_content(is_id: int, cat_id: int, language: str='en', currency: str='USD', country: str='US', toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Get children categories nested in a root category"
    id: The value of 'id' field returned in .../navigations/get-root endpoint
        cat_id: The value of 'cat_id' field returned in .../navigations/get-tabs endpoint
        language: The 2-letter language code
        currency: The 3-letter currency code
        country: The 2-letter country code
        
    """
    url = f"https://unofficial-shein.p.rapidapi.com/navigations/get-node-content"
    querystring = {'id': is_id, 'cat_id': cat_id, }
    if language:
        querystring['language'] = language
    if currency:
        querystring['currency'] = currency
    if country:
        querystring['country'] = country
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "unofficial-shein.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def navigations_get_root(channeltype: int, currency: str='USD', country: str='US', language: str='en', toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Get root categories related to a tab"
    channeltype: The value of 'id' field returned in .../navigations/get-tabs endpoint
        currency: The 3-letter currency code
        country: The 2-letter country code
        language: The 2-letter language code
        
    """
    url = f"https://unofficial-shein.p.rapidapi.com/navigations/get-root"
    querystring = {'channelType': channeltype, }
    if currency:
        querystring['currency'] = currency
    if country:
        querystring['country'] = country
    if language:
        querystring['language'] = language
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "unofficial-shein.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def navigations_get_tabs(language: str='en', country: str='US', currency: str='USD', toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Get tabs for navigation"
    language: The 2-letter language code
        country: The 2-letter country code
        currency: The 3-letter currency code
        
    """
    url = f"https://unofficial-shein.p.rapidapi.com/navigations/get-tabs"
    querystring = {}
    if language:
        querystring['language'] = language
    if country:
        querystring['country'] = country
    if currency:
        querystring['currency'] = currency
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "unofficial-shein.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def countries_detail(region: str='US', toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Get detailed information of a country"
    region: The value of 'value' field returned in .../countries/list endpoint
        
    """
    url = f"https://unofficial-shein.p.rapidapi.com/countries/detail"
    querystring = {}
    if region:
        querystring['region'] = region
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "unofficial-shein.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def countries_list(toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "List all available and supported countries. This endpoint provides meta data for other endpoints."
    
    """
    url = f"https://unofficial-shein.p.rapidapi.com/countries/list"
    querystring = {}
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "unofficial-shein.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def auto_complete(word: str, currency: str='USD', country: str='US', language: str='en', toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Get suggestions by term or phrase"
    word: Any term or phrase that you are familiar with
        currency: The 3-letter currency code
        country: The 2-letter country code
        language: The 2-letter language code
        
    """
    url = f"https://unofficial-shein.p.rapidapi.com/auto-complete"
    querystring = {'word': word, }
    if currency:
        querystring['currency'] = currency
    if country:
        querystring['country'] = country
    if language:
        querystring['language'] = language
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "unofficial-shein.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

