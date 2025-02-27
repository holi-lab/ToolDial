import requests
import json
from datetime import date, datetime, timedelta
import os

from typing import Optional, Dict, Union, List


def products_v3_get_details(tcin: int, store_id: int, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Get detail information of product"
    tcin: The value of tcin field returned in .../products/v2/list or .../products/search-by-barcode endpoint
        store_id: The value of location_id returned in .../stores/list endpoint
        
    """
    url = f"https://target1.p.rapidapi.com/products/v3/get-details"
    querystring = {'tcin': tcin, 'store_id': store_id, }
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "target1.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def products_search_by_barcode(store_id: str, barcode: str, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Search product by barcode"
    store_id: The value of location_id returned in …/stores/list endpoint
        barcode: The barcode 
        
    """
    url = f"https://target1.p.rapidapi.com/products/search-by-barcode"
    querystring = {'store_id': store_id, 'barcode': barcode, }
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "target1.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def categories_list_deprecated(category: str=None, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "List all root and sub categories"
    category: Used to list child categories, you need to parse the value of target field returned right in this endpoint, such as : ...?category=5xtg6
        
    """
    url = f"https://target1.p.rapidapi.com/categories/list"
    querystring = {}
    if category:
        querystring['category'] = category
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "target1.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def categories_v2_list(category: str=None, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "List all root and sub categories"
    category: Used to list child categories, you need to parse the value of target field (taxonomy_nodes->actions->target) OR children->node_id returned right in this endpoint, such as : ...?category=5xtg6
        
    """
    url = f"https://target1.p.rapidapi.com/categories/v2/list"
    querystring = {}
    if category:
        querystring['category'] = category
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "target1.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def products_v2_list(category: str, store_id: str, offset: int=0, faceted_value: str=None, sort_by: str='relevance', keyword: str=None, default_purchasability_filter: bool=None, count: int=20, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "List products in specific store with options and filters"
    category: You need to parse the value of target field returned in .../categories/list endpoint, such as : ...?category=o9rnh. Please notice that do NOT use keyword and category parameters together, keyword  OR category  is required.
        store_id: The value of location_id returned in .../stores/list endpoint
        offset: For paging purpose
        faceted_value: Look for suitable values returned under facet_list/details/value or facet_list/details/facet_id JSON object, separated by comma for multiple options, such as : 5tal2,q643lesaelr,etc...
        sort_by: One of the following is allowed relevance|newest|RatingHigh|bestselling|PriceLow|PriceHigh
        keyword: Search for products by term or phrase, such as : macbook air. Please notice that do NOT use searchTerm and endecaId parameters together, searchTerm  OR endecaId  is required.
        default_purchasability_filter: Filter for purchasable items only
        count: For paging purpose, maximum 20 items per page.
        
    """
    url = f"https://target1.p.rapidapi.com/products/v2/list"
    querystring = {'category': category, 'store_id': store_id, }
    if offset:
        querystring['offset'] = offset
    if faceted_value:
        querystring['faceted_value'] = faceted_value
    if sort_by:
        querystring['sort_by'] = sort_by
    if keyword:
        querystring['keyword'] = keyword
    if default_purchasability_filter:
        querystring['default_purchasability_filter'] = default_purchasability_filter
    if count:
        querystring['count'] = count
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "target1.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def products_list_collection_deprecated(store_id: int, tcin: int, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "List whole collection relating to a product"
    store_id: The value of location_id returned in .../stores/list endpoint
        tcin: The value of tcin field returned in .../products/list endpoint
        
    """
    url = f"https://target1.p.rapidapi.com/products/list-collection"
    querystring = {'store_id': store_id, 'tcin': tcin, }
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "target1.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def products_list_recommended_deprecated(store_id: int, tcins: int, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "List more products to consider"
    store_id: The value of location_id returned in .../stores/list endpoint
        tcins: The value of tcin field returned in .../products/list endpoint
        
    """
    url = f"https://target1.p.rapidapi.com/products/list-recommended"
    querystring = {'store_id': store_id, 'tcins': tcins, }
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "target1.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def products_list_deprecated(endecaid: str, storeid: int, storesearch: bool=None, facets: str=None, searchterm: str=None, pagesize: int=20, sortby: str='relevance', pagenumber: int=1, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "List products in specific store with options and filters"
    endecaid: You need to parse the value of target field returned in .../categories/list endpoint, such as : ...?category=o9rnh. Please notice that do NOT use searchTerm and endecaId parameters together, searchTerm  OR endecaId  is required.
        storeid: The value of location_id returned in .../stores/list endpoint
        storesearch: Only search for In-store products
        facets: Look for suitable values returned under facetView/Entry/ExtendedData/value JSON object, separated by comma for multiple options, such as : 5tal2,q643lesaelr,etc...
        searchterm: Search for products by term or phrase, such as : macbook air. Please notice that do NOT use searchTerm and endecaId parameters together, searchTerm  OR endecaId  is required.
        pagesize: For paging purpose, maximum 20 items per page.
        sortby: One of the following is allowed relevance|newest|RatingHigh|bestselling|PriceLow|PriceHigh
        pagenumber: For paging purpose
        
    """
    url = f"https://target1.p.rapidapi.com/products/list"
    querystring = {'endecaId': endecaid, 'storeId': storeid, }
    if storesearch:
        querystring['storeSearch'] = storesearch
    if facets:
        querystring['facets'] = facets
    if searchterm:
        querystring['searchTerm'] = searchterm
    if pagesize:
        querystring['pageSize'] = pagesize
    if sortby:
        querystring['sortBy'] = sortby
    if pagenumber:
        querystring['pageNumber'] = pagenumber
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "target1.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def products_v2_list_recommended(store_id: int, tcins: int, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "List more products to consider"
    store_id: The value of location_id returned in .../stores/list endpoint
        tcins: The value of tcin field returned in .../products/list endpoint
        
    """
    url = f"https://target1.p.rapidapi.com/products/v2/list-recommended"
    querystring = {'store_id': store_id, 'tcins': tcins, }
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "target1.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def products_v2_get_details_deprecated(store_id: int, tcin: int, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Get detail information of product"
    store_id: The value of location_id returned in .../stores/list endpoint
        tcin: The value of tcin field returned in .../products/list endpoint
        
    """
    url = f"https://target1.p.rapidapi.com/products/v2/get-details"
    querystring = {'store_id': store_id, 'tcin': tcin, }
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "target1.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def products_get_details_deprecated(tcin: int, store_id: int, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Get detail information of product"
    tcin: The value of tcin field returned in .../products/list endpoint
        store_id: The value of location_id returned in .../stores/list endpoint
        
    """
    url = f"https://target1.p.rapidapi.com/products/get-details"
    querystring = {'tcin': tcin, 'store_id': store_id, }
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "target1.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def reviews_v2_list(reviewedid: int, sortby: str='most_recent', page: int=0, hasonlyphotos: bool=None, verifiedonly: bool=None, ratingfilter: str=None, size: int=30, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "List reviews relating to a product"
    reviewedid: The value of tcin field returned in .../products/v2/list endpoint
        sortby: One of the following is allowed most&#95;recent|helpfulness&#95;desc|highest&#95;rating|lowest&#95;rating
        page: For paging purpose
        hasonlyphotos: Only filter for reviews having photos 
        verifiedonly: Only filter for verified reviews 
        ratingfilter: One of the following : rating&#95;1|rating&#95;2|...|rating&#95;5. Pass this parameter multiple time to filter by multiple ratings. Ex : ...&ratingFilter=rating&#95;4&ratingFilter=rating&#95;5&...
        size: For paging purpose, maximum items per page is 30
        
    """
    url = f"https://target1.p.rapidapi.com/reviews/v2/list"
    querystring = {'reviewedId': reviewedid, }
    if sortby:
        querystring['sortBy'] = sortby
    if page:
        querystring['page'] = page
    if hasonlyphotos:
        querystring['hasOnlyPhotos'] = hasonlyphotos
    if verifiedonly:
        querystring['verifiedOnly'] = verifiedonly
    if ratingfilter:
        querystring['ratingFilter'] = ratingfilter
    if size:
        querystring['size'] = size
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "target1.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def stores_get_details(location_id: int, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Get detail information of specific store"
    location_id: The value of location_id returned in .../stores/list endpoint
        
    """
    url = f"https://target1.p.rapidapi.com/stores/get-details"
    querystring = {'location_id': location_id, }
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "target1.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def stores_list(zipcode: str, city: str=None, latlng: str=None, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "List available nearby store by zipcode, GEO location, or city name. One at a time, do NOT use them together at once."
    zipcode: Zip code or postal code of area to look for stores, such as : 10009
        city: Name of city to look for stores, such as : california
        latlng: GEO location to look for around stores, such as : 36.839828,-119.727711
        
    """
    url = f"https://target1.p.rapidapi.com/stores/list"
    querystring = {'zipcode': zipcode, }
    if city:
        querystring['city'] = city
    if latlng:
        querystring['latlng'] = latlng
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "target1.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def auto_complete(q: str, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Get suggestion by term or phrase"
    q: Any familiar term or phrase  to search for products
        
    """
    url = f"https://target1.p.rapidapi.com/auto-complete"
    querystring = {'q': q, }
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "target1.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def reviews_list_deprecated(tcin: int, offset: int=0, limit: int=30, sort: str='time_desc', toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "List reviews relating to a product"
    tcin: The value of tcin field returned in .../products/list endpoint
        offset: For paging purpose
        limit: For paging purpose, maximum items per page is 30
        sort: One of the following is allowed time&#95;desc|helpfulness&#95;desc|rating&#95;desc|rating&#95;asc
        
    """
    url = f"https://target1.p.rapidapi.com/reviews/list"
    querystring = {'tcin': tcin, }
    if offset:
        querystring['offset'] = offset
    if limit:
        querystring['limit'] = limit
    if sort:
        querystring['sort'] = sort
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "target1.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

