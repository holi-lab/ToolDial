import requests
import json
from datetime import date, datetime, timedelta
import os

from typing import Optional, Dict, Union, List


def us_products_v2_detail(productid: str, preferedsku: str, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Get details of product by id"
    productid: The value of productId returned in .../products/list or .../products/search endpoint
        preferedsku: The value of skuId field returned in .../products/list or .../products/search endpoint
        
    """
    url = f"https://sephora.p.rapidapi.com/us/products/v2/detail"
    querystring = {'productId': productid, 'preferedSku': preferedsku, }
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "sephora.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def us_products_v2_search(q: str, pagesize: int=60, ref: str=None, currentpage: int=1, sortby: str=None, pl: int=None, node: str=None, ph: int=None, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Search products by term or phrase with options and filters"
    q: Any term or phrase to search for relating products
        pagesize: For paging purpose
        ref: The value of refinementValue fields returned right in this endpoint. Pass this parameter several times for multiple filters. Ex : ...&ref=filters[Brand]=Shark Beauty&ref=filters[Brand]=ghd&...
        currentpage: For paging purpose
        sortby: One of the following : P&#95;BEST&#95;SELLING:0|P&#95;BEST&#95;SELLING:1|P&#95;PROD&#95;NAME:0|P&#95;PROD&#95;NAME:1|P&#95;NEW:1|P&#95;NEW:0|P&#95;START&#95;DATE:1|P&#95;START&#95;DATE:0|P&#95;RATING:0|P&#95;RATING:1|P&#95;SEPH&#95;EXCLUSIVE:1|P&#95;SEPH&#95;EXCLUSIVE:0|price:1|price:00 or 1 value means asc or desc

        pl: Filter by min price. Check \\\\\\\"price range\\\\\\\" under refinements JSON object for suitable price range.
        node: The value of categoryId returned in .../categories/list or .../categories/list-root
        ph: Filter by max price. Check \\\\\\\"price range\\\\\\\" under refinements JSON object for suitable price range.
        
    """
    url = f"https://sephora.p.rapidapi.com/us/products/v2/search"
    querystring = {'q': q, }
    if pagesize:
        querystring['pageSize'] = pagesize
    if ref:
        querystring['ref'] = ref
    if currentpage:
        querystring['currentPage'] = currentpage
    if sortby:
        querystring['sortBy'] = sortby
    if pl:
        querystring['pl'] = pl
    if node:
        querystring['node'] = node
    if ph:
        querystring['ph'] = ph
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "sephora.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def us_products_v2_list(categoryid: str, ref: str=None, pagesize: int=60, currentpage: int=1, sortby: str=None, ph: int=None, pl: int=None, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "List products with options and filters"
    categoryid: The value of categoryId returned in .../categories/list or .../categories/list-root
        ref: The value of refinementValue fields returned right in this endpoint. Pass this parameter several times for multiple filters. Ex : ...&ref=filters[Brand]=Shark Beauty&ref=filters[Brand]=ghd&...
        pagesize: For paging purpose
        currentpage: For paging purpose
        sortby: One of the following : P&#95;BEST&#95;SELLING:0|P&#95;BEST&#95;SELLING:1|P&#95;PROD&#95;NAME:0|P&#95;PROD&#95;NAME:1|P&#95;NEW:1|P&#95;NEW:0|P&#95;START&#95;DATE:1|P&#95;START&#95;DATE:0|P&#95;RATING:0|P&#95;RATING:1|P&#95;SEPH&#95;EXCLUSIVE:1|P&#95;SEPH&#95;EXCLUSIVE:0|price:1|price:0
0 or 1 value means asc or desc
        ph: Filter by max price. Check \\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\"price range\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\" under refinements JSON object for suitable price range.
        pl: Filter by min price. Check \\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\"price range\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\" under refinements JSON object for suitable price range.
        
    """
    url = f"https://sephora.p.rapidapi.com/us/products/v2/list"
    querystring = {'categoryId': categoryid, }
    if ref:
        querystring['ref'] = ref
    if pagesize:
        querystring['pageSize'] = pagesize
    if currentpage:
        querystring['currentPage'] = currentpage
    if sortby:
        querystring['sortBy'] = sortby
    if ph:
        querystring['ph'] = ph
    if pl:
        querystring['pl'] = pl
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "sephora.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def reviews_list(productid: str, ratingvalue: int=None, offset: int=0, limit: int=60, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "List reviews relating to specific product"
    ratingvalue: Filter by rating score from 1 to 5
        offset: For paging purpose
        limit: For paging purpose
        
    """
    url = f"https://sephora.p.rapidapi.com/reviews/list"
    querystring = {'ProductId': productid, }
    if ratingvalue:
        querystring['RatingValue'] = ratingvalue
    if offset:
        querystring['Offset'] = offset
    if limit:
        querystring['Limit'] = limit
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "sephora.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def products_check_availability(latitude: int, skuid: str, longitude: int, radius: int=25, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Check availability of specific product around a location"
    latitude: Latitude of GEO location to search for nearest stocked stores
        skuid: The value of skuId field returned in .../products/list or .../products/search endpoint
        longitude: Longitude of GEO location to search for nearest stocked stores
        radius: The radious around the GEO location to search for stores
        
    """
    url = f"https://sephora.p.rapidapi.com/products/check-availability"
    querystring = {'latitude': latitude, 'skuId': skuid, 'longitude': longitude, }
    if radius:
        querystring['radius'] = radius
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "sephora.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def products_search_by_barcode(upccode: str, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Search product by barcode (UPC)"
    upccode: The scanned code (UPC)
        
    """
    url = f"https://sephora.p.rapidapi.com/products/search-by-barcode"
    querystring = {'upccode': upccode, }
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "sephora.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def products_search_deprecated(q: str, sortby: str=None, pagesize: int=60, pl: int=None, ref: str=None, currentpage: int=1, node: str=None, ph: int=None, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Search products by term or phrase with options and filters"
    q: Any term or phrase to search for relating products
        sortby: One of the following : P&#95;BEST&#95;SELLING:0|P&#95;BEST&#95;SELLING:1|P&#95;PROD&#95;NAME:0|P&#95;PROD&#95;NAME:1|P&#95;NEW:1|P&#95;NEW:0|P&#95;START&#95;DATE:1|P&#95;START&#95;DATE:0|P&#95;RATING:0|P&#95;RATING:1|P&#95;SEPH&#95;EXCLUSIVE:1|P&#95;SEPH&#95;EXCLUSIVE:0|price:1|price:00 or 1 value means asc or desc

        pagesize: For paging purpose
        pl: Filter by min price. Check \\\\\\\"price range\\\\\\\" under refinements JSON object for suitable price range.
        ref: The value of refinementValueId fields returned right in this endpoint. Pass this parameter several times for multiple filters. Ex : ...&ref=21972988&ref=5024296&...
        currentpage: For paging purpose
        node: The value of categoryId returned in .../categories/list or .../categories/list-root
        ph: Filter by max price. Check \\\\\\\"price range\\\\\\\" under refinements JSON object for suitable price range.
        
    """
    url = f"https://sephora.p.rapidapi.com/products/search"
    querystring = {'q': q, }
    if sortby:
        querystring['sortBy'] = sortby
    if pagesize:
        querystring['pageSize'] = pagesize
    if pl:
        querystring['pl'] = pl
    if ref:
        querystring['ref'] = ref
    if currentpage:
        querystring['currentPage'] = currentpage
    if node:
        querystring['node'] = node
    if ph:
        querystring['ph'] = ph
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "sephora.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def products_list_deprecated(categoryid: str, currentpage: int=1, ref: str=None, pl: int=None, ph: int=None, sortby: str=None, pagesize: int=60, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "List products with options and filters"
    categoryid: The value of categoryId returned in .../categories/list or .../categories/list-root
        currentpage: For paging purpose
        ref: The value of refinementValueId fields returned right in this endpoint. Pass this parameter several times for multiple filters. Ex : ...&ref=21972988&ref=5024296&...
        pl: Filter by min price. Check \\\\\\\\\\\\\\\"price range\\\\\\\\\\\\\\\" under refinements JSON object for suitable price range.
        ph: Filter by max price. Check \\\\\\\\\\\\\\\"price range\\\\\\\\\\\\\\\" under refinements JSON object for suitable price range.
        sortby: One of the following : P&#95;BEST&#95;SELLING:0|P&#95;BEST&#95;SELLING:1|P&#95;PROD&#95;NAME:0|P&#95;PROD&#95;NAME:1|P&#95;NEW:1|P&#95;NEW:0|P&#95;START&#95;DATE:1|P&#95;START&#95;DATE:0|P&#95;RATING:0|P&#95;RATING:1|P&#95;SEPH&#95;EXCLUSIVE:1|P&#95;SEPH&#95;EXCLUSIVE:0|price:1|price:0
0 or 1 value means asc or desc
        pagesize: For paging purpose
        
    """
    url = f"https://sephora.p.rapidapi.com/products/list"
    querystring = {'categoryId': categoryid, }
    if currentpage:
        querystring['currentPage'] = currentpage
    if ref:
        querystring['ref'] = ref
    if pl:
        querystring['pl'] = pl
    if ph:
        querystring['ph'] = ph
    if sortby:
        querystring['sortBy'] = sortby
    if pagesize:
        querystring['pageSize'] = pagesize
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "sephora.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def categories_list(categoryid: str, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "List categories and its recursive."
    categoryid: The value of categoryId returned right in this endpoint or .../categories/list-root endpoint
        
    """
    url = f"https://sephora.p.rapidapi.com/categories/list"
    querystring = {'categoryId': categoryid, }
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "sephora.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def reviews_v2_list(is_id: int, country: str='SG', sort: str='recent', size: int=10, number: int=1, native: bool=None, variants: int=None, language: str='en-SG', toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "List reviews related to a product"
    id: The value of id field returned in .../products/v2/list endpoint.
        country: One of the following : HK|AU|BN|ID|MY|NZ|PH|SG|TW|TH
        sort: One of the following : recent|-recent
        size: The number of items per response for paging purpose
        number: The page index for paging purpose
        native: true - Exclude Sephora US reviews
        variants: The value of variant -> data -> id field returned right in this endpoint. To filter reviews related to specific product variant.
        language: One of the following and need to be pair with country parameter properly: en-HK|zh-HK|en-AU|en-BN|id-ID|en-ID|en-MY|en-NZ|en-PH|en-SG|en-TW|zh-TW|th-TH|en-TH
        
    """
    url = f"https://sephora.p.rapidapi.com/reviews/v2/list"
    querystring = {'id': is_id, }
    if country:
        querystring['country'] = country
    if sort:
        querystring['sort'] = sort
    if size:
        querystring['size'] = size
    if number:
        querystring['number'] = number
    if native:
        querystring['native'] = native
    if variants:
        querystring['variants'] = variants
    if language:
        querystring['language'] = language
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "sephora.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def products_v2_detail(is_id: int, country: str='SG', language: str='en-SG', toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Get detailed information of a product"
    id: The value of id field returned in .../products/v2/list endpoint.
        country: One of the following : HK|AU|BN|ID|MY|NZ|PH|SG|TW|TH
        language: One of the following and need to be pair with country parameter properly: en-HK|zh-HK|en-AU|en-BN|id-ID|en-ID|en-MY|en-NZ|en-PH|en-SG|en-TW|zh-TW|th-TH|en-TH
        
    """
    url = f"https://sephora.p.rapidapi.com/products/v2/detail"
    querystring = {'id': is_id, }
    if country:
        querystring['country'] = country
    if language:
        querystring['language'] = language
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "sephora.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def products_v2_list(number: int=1, query: str=None, size: int=30, country: str='SG', sort: str='sales', brand: str=None, product_group: str=None, min_price: int=None, others: str=None, category: str=None, max_price: int=None, root_brand: str=None, root_category: str=None, language: str='en-SG', filter_type: str=None, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "List products with options and filters"
    number: The page index for paging purpose
        query: Search product by term. Ex : bikini top
        size: The number of items per response for paging purpose
        country: One of the following : HK|AU|BN|ID|MY|NZ|PH|SG|TW|TH
        sort: One of the following : relevance|sales|published_at|rating|price|-price
        brand: The value of \\\\\\\"id\\\\\\\" field returned in .../brands/v2/list. Separated by comma for multiple options. Ex : 266,2636
        product_group: One of the following : new-arrivals|bestsellers
        min_price: Filter products by price
        others: Support only value : on_sale
        category: The value of \\\"id\\\" field returned in .../categories/v2/list endpoint. Separated by comma for multiple options. Ex : 801,805
        max_price: Filter products by price
        root_brand: The value of \\\\\\\"slug-url\\\\\\\" field returned in .../brands/v2/list endpoint. Ex : aerin
        root_category: The value of \\\\\\\"slug-url\\\\\\\" field returned in .../categories/v2/list. Ex : makeup
        language: One of the following and need to be pair with country parameter properly: en-HK|zh-HK|en-AU|en-BN|id-ID|en-ID|en-MY|en-NZ|en-PH|en-SG|en-TW|zh-TW|th-TH|en-TH
        filter_type: Filter products by types, the format is {id of filter-type}_{id of filter-value} returned in  .../products/v2/get-filters endpoint. Separated by comma for multiple options. Ex : 41_684,41_686,38_714,38_504
        
    """
    url = f"https://sephora.p.rapidapi.com/products/v2/list"
    querystring = {}
    if number:
        querystring['number'] = number
    if query:
        querystring['query'] = query
    if size:
        querystring['size'] = size
    if country:
        querystring['country'] = country
    if sort:
        querystring['sort'] = sort
    if brand:
        querystring['brand'] = brand
    if product_group:
        querystring['product_group'] = product_group
    if min_price:
        querystring['min_price'] = min_price
    if others:
        querystring['others'] = others
    if category:
        querystring['category'] = category
    if max_price:
        querystring['max_price'] = max_price
    if root_brand:
        querystring['root_brand'] = root_brand
    if root_category:
        querystring['root_category'] = root_category
    if language:
        querystring['language'] = language
    if filter_type:
        querystring['filter_type'] = filter_type
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "sephora.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def categories_v2_list(country: str='SG', size: int=100, number: int=1, language: str='en-SG', toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "List categories"
    country: One of the following : HK|AU|BN|ID|MY|NZ|PH|SG|TW|TH
        size: The number of items per response for paging purpose
        number: The page index for paging purpose
        language: One of the following and need to be pair with country parameter properly: en-HK|zh-HK|en-AU|en-BN|id-ID|en-ID|en-MY|en-NZ|en-PH|en-SG|en-TW|zh-TW|th-TH|en-TH
        
    """
    url = f"https://sephora.p.rapidapi.com/categories/v2/list"
    querystring = {}
    if country:
        querystring['country'] = country
    if size:
        querystring['size'] = size
    if number:
        querystring['number'] = number
    if language:
        querystring['language'] = language
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "sephora.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def brands_v2_list(size: int=100, number: int=1, country: str='SG', language: str='en-SG', toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "List brands"
    size: The number of items per response for paging purpose
        number: The page index for paging purpose
        country: One of the following : HK|AU|BN|ID|MY|NZ|PH|SG|TW|TH
        language: One of the following and need to be pair with country parameter properly: en-HK|zh-HK|en-AU|en-BN|id-ID|en-ID|en-MY|en-NZ|en-PH|en-SG|en-TW|zh-TW|th-TH|en-TH
        
    """
    url = f"https://sephora.p.rapidapi.com/brands/v2/list"
    querystring = {}
    if size:
        querystring['size'] = size
    if number:
        querystring['number'] = number
    if country:
        querystring['country'] = country
    if language:
        querystring['language'] = language
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "sephora.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def v2_auto_complete(query: str, country: str='SG', language: str='en-SG', toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Get suggestions by term or phrase"
    query: Any term or phrase that you are familiar with
        country: One of the following : HK|AU|BN|ID|MY|NZ|PH|SG|TW|TH
        language: One of the following and need to be pair with country parameter properly: en-HK|zh-HK|en-AU|en-BN|id-ID|en-ID|en-MY|en-NZ|en-PH|en-SG|en-TW|zh-TW|th-TH|en-TH
        
    """
    url = f"https://sephora.p.rapidapi.com/v2/auto-complete"
    querystring = {'query': query, }
    if country:
        querystring['country'] = country
    if language:
        querystring['language'] = language
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "sephora.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def stores_list(longitude: int, latitude: int, radius: int=25, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "List nearest stores around a GEO location"
    longitude: Longitude of GEO location to search for nearest stores
        latitude: Latitude of GEO location to search for nearest stores
        radius: Radius to look for around stores
        
    """
    url = f"https://sephora.p.rapidapi.com/stores/list"
    querystring = {'longitude': longitude, 'latitude': latitude, }
    if radius:
        querystring['radius'] = radius
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "sephora.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def auto_complete(q: str, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Get suggestions by term or phrase"
    q: Any term or phrase that you are familiar with
        
    """
    url = f"https://sephora.p.rapidapi.com/auto-complete"
    querystring = {'q': q, }
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "sephora.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def products_v2_search_by_barcode(upcs: str, country: str='SG', language: str='en-SG', toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Search product by upc"
    upcs: The UPC barcode
        country: One of the following : HK|AU|BN|ID|MY|NZ|PH|SG|TW|TH
        language: One of the following and need to be pair with country parameter properly: en-HK|zh-HK|en-AU|en-BN|id-ID|en-ID|en-MY|en-NZ|en-PH|en-SG|en-TW|zh-TW|th-TH|en-TH
        
    """
    url = f"https://sephora.p.rapidapi.com/products/v2/search-by-barcode"
    querystring = {'upcs': upcs, }
    if country:
        querystring['country'] = country
    if language:
        querystring['language'] = language
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "sephora.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def products_detail_deprecated(productid: str, preferedsku: str, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Get details of product by id"
    productid: The value of productId returned in .../products/list or .../products/search endpoint
        preferedsku: The value of skuId field returned in .../products/list or .../products/search endpoint
        
    """
    url = f"https://sephora.p.rapidapi.com/products/detail"
    querystring = {'productId': productid, 'preferedSku': preferedsku, }
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "sephora.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def categories_list_root_deprecated(toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "List top root categories"
    
    """
    url = f"https://sephora.p.rapidapi.com/categories/list-root"
    querystring = {}
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "sephora.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def categories_v2_list_root(toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "List top root categories"
    
    """
    url = f"https://sephora.p.rapidapi.com/categories/v2/list-root"
    querystring = {}
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "sephora.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def products_v2_get_filters(category: str=None, country: str='SG', brand: str=None, others: str=None, max_price: int=None, search: str=None, product_group: str=None, min_price: int=None, language: str='en-SG', filter_type: str=None, root_brand: str='aerin', root_category: str='makeup', toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Generate the filter panel dynamically"
    category: The value of \\\"id\\\" field returned in .../categories/v2/list endpoint. Separated by comma for multiple options. Ex : 801,805
        country: One of the following : HK|AU|BN|ID|MY|NZ|PH|SG|TW|TH
        brand: The value of \\\\\\\"id\\\\\\\" field returned in .../brands/v2/list. Separated by comma for multiple options. Ex : 266,2636
        others: Support only value : on_sale
        max_price: Filter products by price
        search: Search product by term. Ex : bikini top
        product_group: One of the following : new-arrivals|bestsellers
        min_price: Filter products by price
        language: One of the following and need to be pair with country parameter properly: en-HK|zh-HK|en-AU|en-BN|id-ID|en-ID|en-MY|en-NZ|en-PH|en-SG|en-TW|zh-TW|th-TH|en-TH
        filter_type: Filter products by types, the format is {id of filter-type}_{id of filter-value} returned in  .../products/v2/get-filters endpoint. Separated by comma for multiple options. Ex : 41_684,41_686,38_714,38_504
        root_brand: Either root_brand or root_category is required. The value of \"slug-url\" field returned in .../brands/v2/list endpoint. Ex : aerin
        root_category: Either root_brand or root_category is required. The value of \"slug-url\" field returned in .../categories/v2/list. Ex : makeup
        
    """
    url = f"https://sephora.p.rapidapi.com/products/v2/get-filters"
    querystring = {}
    if category:
        querystring['category'] = category
    if country:
        querystring['country'] = country
    if brand:
        querystring['brand'] = brand
    if others:
        querystring['others'] = others
    if max_price:
        querystring['max_price'] = max_price
    if search:
        querystring['search'] = search
    if product_group:
        querystring['product_group'] = product_group
    if min_price:
        querystring['min_price'] = min_price
    if language:
        querystring['language'] = language
    if filter_type:
        querystring['filter_type'] = filter_type
    if root_brand:
        querystring['root_brand'] = root_brand
    if root_category:
        querystring['root_category'] = root_category
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "sephora.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

