import requests
import json
from datetime import date, datetime, timedelta
import os

from typing import Optional, Dict, Union, List


def item_info(itemid: int, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Retrieves original item information from Jingdong App"
    itemid: Jingdong Item Identifier
        
    """
    url = f"https://jingdong-Jing-Dong-data-service.p.rapidapi.com/item/itemInfo"
    querystring = {'itemId': itemid, }
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "jingdong-Jing-Dong-data-service.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def seller_items(minvendorrating: int=None, maxprice: int=None, target_language: str=None, size: int=None, sellerid: str=None, minprice: int=None, maxvendorrating: int=None, page: int=None, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Retrieves a list of all seller items"
    minvendorrating: Minimum Item Vendor Rating
        maxprice: Maximum Item Price
        target_language: The language of translation, list of all supported languages can be found [here](https://rapidapi.com/iamEvara/api/jingdong-Jing-Dong-data-service/tutorials/list-of-all-supported-languages).
        size: Number of results per page.
        sellerid: Jingdong Seller Identifier
        minprice: Minimum Item Price
        maxvendorrating: Maximum Item Vendor Rating
        page: The page number of the results to be retrieved. Default is 1.
        
    """
    url = f"https://jingdong-Jing-Dong-data-service.p.rapidapi.com/seller/sellerItems"
    querystring = {}
    if minvendorrating:
        querystring['minVendorRating'] = minvendorrating
    if maxprice:
        querystring['maxPrice'] = maxprice
    if target_language:
        querystring['target_language'] = target_language
    if size:
        querystring['size'] = size
    if sellerid:
        querystring['sellerId'] = sellerid
    if minprice:
        querystring['minPrice'] = minprice
    if maxvendorrating:
        querystring['maxVendorRating'] = maxvendorrating
    if page:
        querystring['page'] = page
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "jingdong-Jing-Dong-data-service.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def seller_info(sellerid: str, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Retrieves detailed seller info"
    sellerid: Jingdong Seller Identifier
        
    """
    url = f"https://jingdong-Jing-Dong-data-service.p.rapidapi.com/seller/sellerInfo"
    querystring = {'sellerId': sellerid, }
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "jingdong-Jing-Dong-data-service.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def search_seller_items(sellerid: str, query: str, sort: str=None, page: int=None, minsellerrating: int=None, maxsellerrating: int=None, size: int=None, target_language: str=None, maxprice: int=None, minprice: int=None, query_language: str=None, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Searches for sellers’ items using a query string"
    sellerid: Jingdong Seller Identifier
        query: Search query
        sort: This parameter is used to specify the sorting order of the items returned in the API response. There are six available sorting options for the `sort` parameter:

`default`: This is the default sort option, and it sorts the items based on the Pinduoduo platform's default sort order.

`price_asc`: This option sorts the items in ascending order of their prices, with the lowest priced items appearing first in the response.

`price_desc`: This option sorts the items in descending order of their prices, with the highest priced items appearing first in the response.

`total_price_asc`: This option sorts the items in ascending order of their total prices, which includes any discounts or promotions, with the lowest total priced items appearing first in the response.

`total_price_desc`: This option sorts the items in descending order of their total prices, with the highest total priced items appearing first in the response.

`volume_desc`: This option sorts the items in descending order of their sales volume, with the best-selling items appearing first in the response.

`vendor_rating_desc`: This option sorts the items in descending order of their seller rating, with the highest rated sellers appearing first in the response.

`updated_time_desc`: This option sorts the items in descending order of their update time, with the most recently updated items appearing first in the response.

        page: The page number of the results to be retrieved. Default is 1.
        minsellerrating: Minimum Item Seller Rating
        maxsellerrating: Maximum Item Seller Rating
        size: Number of results per page.
        target_language: The language of translation, list of all supported languages can be found [here](https://rapidapi.com/iamEvara/api/jingdong-Jing-Dong-data-service/tutorials/list-of-all-supported-languages).
        maxprice: Maximum Item Price
        minprice: Minimum Item Price
        query_language: The `query_language` parameter specifies the language of the search query provided in the `query` parameter for translation into Chinese. This is important because the query must be in Chinese to be searched on the platform.
        
    """
    url = f"https://jingdong-Jing-Dong-data-service.p.rapidapi.com/seller/searchSellerItems"
    querystring = {'sellerId': sellerid, 'query': query, }
    if sort:
        querystring['sort'] = sort
    if page:
        querystring['page'] = page
    if minsellerrating:
        querystring['minSellerRating'] = minsellerrating
    if maxsellerrating:
        querystring['maxSellerRating'] = maxsellerrating
    if size:
        querystring['size'] = size
    if target_language:
        querystring['target_language'] = target_language
    if maxprice:
        querystring['maxPrice'] = maxprice
    if minprice:
        querystring['minPrice'] = minprice
    if query_language:
        querystring['query_language'] = query_language
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "jingdong-Jing-Dong-data-service.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def items_batch_info(itemids: str, target_language: str=None, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Retrieves batch information for multiple items at once"
    itemids: Specify the IDs of the items for which simple information will be retrieved. The IDs should be provided as a comma-separated list of values (eg. `100043033735,10066990232982`).
        target_language: The language of translation, list of all supported languages can be found [here](https://rapidapi.com/iamEvara/api/jingdong-Jing-Dong-data-service/tutorials/list-of-all-supported-languages).
        
    """
    url = f"https://jingdong-Jing-Dong-data-service.p.rapidapi.com/item/itemsBatchInfo"
    querystring = {'itemIds': itemids, }
    if target_language:
        querystring['target_language'] = target_language
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "jingdong-Jing-Dong-data-service.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def search_suggestions(query: str, query_language: str=None, target_language: str=None, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Provides suggestions for search queries"
    query: Search query
        query_language: The `query_language` parameter specifies the language of the search query provided in the `query` parameter for translation into Chinese. This is important because the query must be in Chinese to be searched on the platform.
        target_language: The language of translation, list of all supported languages can be found [here](https://rapidapi.com/iamEvara/api/jingdong-Jing-Dong-data-service/tutorials/list-of-all-supported-languages).
        
    """
    url = f"https://jingdong-Jing-Dong-data-service.p.rapidapi.com/search/searchSuggestions"
    querystring = {'query': query, }
    if query_language:
        querystring['query_language'] = query_language
    if target_language:
        querystring['target_language'] = target_language
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "jingdong-Jing-Dong-data-service.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def search_items(query: str, hasdiscount: bool=None, query_language: str=None, size: int=None, minvendorrating: int=None, maxprice: int=None, page: int=None, sort: str=None, maxvendorrating: int=None, instock: bool=None, target_language: str=None, minprice: int=None, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Searches for items using a query string"
    query: Search query
        hasdiscount: If set to true, the results will contain only items with discount, if set to false the results will contain only items without a discount. Don't set the parameter so you can get both results.
        query_language: The `query_language` parameter specifies the language of the search query provided in the `query` parameter for translation into Chinese. This is important because the query must be in Chinese to be searched on the platform.
        size: Number of results per page.
        minvendorrating: Minimum Item Vendor Rating
        maxprice: Maximum Item Price
        page: The page number of the results to be retrieved. Default is 1.
        sort: This parameter is used to specify the sorting order of the items returned in the API response. There are six available sorting options for the `sort` parameter:

`default`: This is the default sort option, and it sorts the items based on the Pinduoduo platform's default sort order.

`price_asc`: This option sorts the items in ascending order of their prices, with the lowest priced items appearing first in the response.

`price_desc`: This option sorts the items in descending order of their prices, with the highest priced items appearing first in the response.

`total_price_asc`: This option sorts the items in ascending order of their total prices, which includes any discounts or promotions, with the lowest total priced items appearing first in the response.

`total_price_desc`: This option sorts the items in descending order of their total prices, with the highest total priced items appearing first in the response.

`volume_desc`: This option sorts the items in descending order of their sales volume, with the best-selling items appearing first in the response.

`vendor_rating_desc`: This option sorts the items in descending order of their seller rating, with the highest rated sellers appearing first in the response.

`updated_time_desc`: This option sorts the items in descending order of their update time, with the most recently updated items appearing first in the response.

        maxvendorrating: Maximum Item Vendor Rating
        instock: If set to true, the results will contain only items in stock, if set to false the results will contain only items out of stock. Don't set the parameter so you can get both results.
        target_language: The language of translation, list of all supported languages can be found [here](https://rapidapi.com/iamEvara/api/jingdong-Jing-Dong-data-service/tutorials/list-of-all-supported-languages).
        minprice: Minimum Item Price
        
    """
    url = f"https://jingdong-Jing-Dong-data-service.p.rapidapi.com/search/searchItems"
    querystring = {'query': query, }
    if hasdiscount:
        querystring['hasDiscount'] = hasdiscount
    if query_language:
        querystring['query_language'] = query_language
    if size:
        querystring['size'] = size
    if minvendorrating:
        querystring['minVendorRating'] = minvendorrating
    if maxprice:
        querystring['maxPrice'] = maxprice
    if page:
        querystring['page'] = page
    if sort:
        querystring['sort'] = sort
    if maxvendorrating:
        querystring['maxVendorRating'] = maxvendorrating
    if instock:
        querystring['inStock'] = instock
    if target_language:
        querystring['target_language'] = target_language
    if minprice:
        querystring['minPrice'] = minprice
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "jingdong-Jing-Dong-data-service.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def item_reviews_detailed(itemid: int, size: int=None, page: int=None, target_language: str=None, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Retrieves all item reviews information, but more detailed"
    itemid: Jingdong Item Identifier
        size: Number of results per page.
        page: The page number of the results to be retrieved. Default is 1.
        target_language: The language of translation, list of all supported languages can be found [here](https://rapidapi.com/iamEvara/api/jingdong-Jing-Dong-data-service/tutorials/list-of-all-supported-languages).
        
    """
    url = f"https://jingdong-Jing-Dong-data-service.p.rapidapi.com/item/itemReviewsDetailed"
    querystring = {'itemId': itemid, }
    if size:
        querystring['size'] = size
    if page:
        querystring['page'] = page
    if target_language:
        querystring['target_language'] = target_language
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "jingdong-Jing-Dong-data-service.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def item_reviews(itemid: int, size: int=None, page: int=None, target_language: str=None, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Retrieves all item reviews"
    itemid: Jingdong Item Identifier
        size: Number of results per page.
        page: The page number of the results to be retrieved. Default is 1.
        target_language: The language of translation, list of all supported languages can be found [here](https://rapidapi.com/iamEvara/api/jingdong-Jing-Dong-data-service/tutorials/list-of-all-supported-languages).
        
    """
    url = f"https://jingdong-Jing-Dong-data-service.p.rapidapi.com/item/itemReviews"
    querystring = {'itemId': itemid, }
    if size:
        querystring['size'] = size
    if page:
        querystring['page'] = page
    if target_language:
        querystring['target_language'] = target_language
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "jingdong-Jing-Dong-data-service.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def item_qa(itemid: int, page: int=None, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Retrieves the Q&A section of an item"
    itemid: Jingdong Item Identifier
        page: The page number of the results to be retrieved. Default is 1.
        
    """
    url = f"https://jingdong-Jing-Dong-data-service.p.rapidapi.com/item/itemQA"
    querystring = {'itemId': itemid, }
    if page:
        querystring['page'] = page
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "jingdong-Jing-Dong-data-service.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def item_full_info(itemid: int, target_language: str=None, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Retrieves item full information"
    itemid: Jingdong Item Identifier
        target_language: The language of translation, list of all supported languages can be found [here](https://rapidapi.com/iamEvara/api/jingdong-Jing-Dong-data-service/tutorials/list-of-all-supported-languages).
        
    """
    url = f"https://jingdong-Jing-Dong-data-service.p.rapidapi.com/item/itemFullInfo"
    querystring = {'itemId': itemid, }
    if target_language:
        querystring['target_language'] = target_language
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "jingdong-Jing-Dong-data-service.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

