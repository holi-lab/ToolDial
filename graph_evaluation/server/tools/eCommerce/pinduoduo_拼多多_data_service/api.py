import requests
import json
from datetime import date, datetime, timedelta
import os

from typing import Optional, Dict, Union, List


def item_info_disabled(itemid: int, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Retrieves original item information from Pinduoduo App"
    itemid: Pinduoduo Item Identifier
        
    """
    url = f"https://pinduoduo-Pin-Duo-Duo-data-service1.p.rapidapi.com/item/itemInfo"
    querystring = {'itemId': itemid, }
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "pinduoduo-Pin-Duo-Duo-data-service1.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def coupons_by_theme(themeid: int, target_language: str=None, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Search for items within a specific theme on Pinduoduo platform. The API returns a list of items matching the specified Theme, including their name, price, image URL, and others."
    themeid: Theme identifier, provided by `/coupon/couponsThemes`.
        target_language: The language of translation, list of all supported languages can be found [here](https://rapidapi.com/iamEvara/api/pinduoduo-Pin-Duo-Duo-data-service1/tutorials/list-of-all-supported-languages).
        
    """
    url = f"https://pinduoduo-Pin-Duo-Duo-data-service1.p.rapidapi.com/coupon/couponsByTheme"
    querystring = {'themeId': themeid, }
    if target_language:
        querystring['target_language'] = target_language
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "pinduoduo-Pin-Duo-Duo-data-service1.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def coupons_themes(size: int=None, target_language: str=None, page: int=None, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Retrieves a list of themes on Pinduoduo platform. Returns information such as the theme ID, name, and image URL."
    size: Number of results per page.
        target_language: The language of translation, list of all supported languages can be found [here](https://rapidapi.com/iamEvara/api/pinduoduo-Pin-Duo-Duo-data-service1/tutorials/list-of-all-supported-languages).
        page: The page number of the results to be retrieved. Default is 1.
        
    """
    url = f"https://pinduoduo-Pin-Duo-Duo-data-service1.p.rapidapi.com/coupon/couponsThemes"
    querystring = {}
    if size:
        querystring['size'] = size
    if target_language:
        querystring['target_language'] = target_language
    if page:
        querystring['page'] = page
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "pinduoduo-Pin-Duo-Duo-data-service1.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def item_qa(itemid: str, target_language: str=None, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Retrieves the Question & Answer section of an item."
    itemid: Item Identifier
        target_language: The language of translation, list of all supported languages can be found [here](https://rapidapi.com/iamEvara/api/pinduoduo-Pin-Duo-Duo-data-service1/tutorials/list-of-all-supported-languages).
        
    """
    url = f"https://pinduoduo-Pin-Duo-Duo-data-service1.p.rapidapi.com/item/itemQA"
    querystring = {'itemId': itemid, }
    if target_language:
        querystring['target_language'] = target_language
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "pinduoduo-Pin-Duo-Duo-data-service1.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def item_simple_info(itemid: str, target_language: str=None, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Retrieves simplified information about an item"
    itemid: Item Identifier
        target_language: The language of translation, list of all supported languages can be found [here](https://rapidapi.com/iamEvara/api/pinduoduo-Pin-Duo-Duo-data-service1/tutorials/list-of-all-supported-languages).
        
    """
    url = f"https://pinduoduo-Pin-Duo-Duo-data-service1.p.rapidapi.com/item/itemSimpleInfo"
    querystring = {'itemId': itemid, }
    if target_language:
        querystring['target_language'] = target_language
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "pinduoduo-Pin-Duo-Duo-data-service1.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def items_orders(start_time: str, end_time: str, target_language: str=None, page: int=None, size: int=None, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Retrieves list of orders made in a time frame on the Pinduoduo platform."
    start_time: Starting time range for which to retrieve the orders information. The time should be provided as a timestamp.
        end_time: Ending time range for which to retrieve the orders information. The time should be provided as a timestamp.
        target_language: The language of translation, list of all supported languages can be found [here](https://rapidapi.com/iamEvara/api/pinduoduo-Pin-Duo-Duo-data-service1/tutorials/list-of-all-supported-languages).
        page: The page number of the results to be retrieved. Default is 1.
        size: Number of results per page.
        
    """
    url = f"https://pinduoduo-Pin-Duo-Duo-data-service1.p.rapidapi.com/item/itemsOrders"
    querystring = {'start_time': start_time, 'end_time': end_time, }
    if target_language:
        querystring['target_language'] = target_language
    if page:
        querystring['page'] = page
    if size:
        querystring['size'] = size
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "pinduoduo-Pin-Duo-Duo-data-service1.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def items_batch_info(itemids: str, target_language: str=None, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Retrieves batch information about multiple items"
    itemids: The itemIds parameter is used to specify a list of item identifiers, separated by commas (`,`), for which you wish to retrieve batch information. Each item identifier corresponds to a unique item within the Pinduoduo system. Please note that the maximum number of item identifiers allowed in a single request is 200.
        target_language: The language of translation, list of all supported languages can be found [here](https://rapidapi.com/iamEvara/api/pinduoduo-Pin-Duo-Duo-data-service1/tutorials/list-of-all-supported-languagess).
        
    """
    url = f"https://pinduoduo-Pin-Duo-Duo-data-service1.p.rapidapi.com/item/itemsBatchInfo"
    querystring = {'itemIds': itemids, }
    if target_language:
        querystring['target_language'] = target_language
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "pinduoduo-Pin-Duo-Duo-data-service1.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def search_items(query: str, sort: str=None, query_language: str=None, target_language: str=None, page: int=None, instock: bool=None, size: int=None, maxprice: int=None, hasdiscount: bool=None, minprice: int=None, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Searches for items using a query string"
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

        query_language: The `query_language` parameter specifies the language of the search query provided in the `query` parameter for translation into Chinese. This is important because the query must be in Chinese to be searched on the platform.
        target_language: The language of translation, list of all supported languages can be found [here](https://rapidapi.com/iamEvara/api/pinduoduo-Pin-Duo-Duo-data-service1/tutorials/list-of-all-supported-languages).
        page: The page number of the results to be retrieved. Default is 1.
        instock: If set to true, the results will contain only items in stock, if set to false the results will contain only items out of stock. Don't set the parameter so you can get both results.
        size: Number of results per page.
        maxprice: Maximum Item Price
        hasdiscount: If set to true, the results will contain only items with discount, if set to false the results will contain only items without a discount. Don't set the parameter so you can get both results.
        minprice: Minimum Item Price
        
    """
    url = f"https://pinduoduo-Pin-Duo-Duo-data-service1.p.rapidapi.com/search/searchItems"
    querystring = {'query': query, }
    if sort:
        querystring['sort'] = sort
    if query_language:
        querystring['query_language'] = query_language
    if target_language:
        querystring['target_language'] = target_language
    if page:
        querystring['page'] = page
    if instock:
        querystring['inStock'] = instock
    if size:
        querystring['size'] = size
    if maxprice:
        querystring['maxPrice'] = maxprice
    if hasdiscount:
        querystring['hasDiscount'] = hasdiscount
    if minprice:
        querystring['minPrice'] = minprice
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "pinduoduo-Pin-Duo-Duo-data-service1.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def search_suggestions(query: str, target_language: str=None, query_language: str=None, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Provides suggestions for search queries"
    query: Search query
        target_language: The language of translation, list of all supported languages can be found [here](https://rapidapi.com/iamEvara/api/pinduoduo-Pin-Duo-Duo-data-service1/tutorials/list-of-all-supported-languages).
        query_language: The `query_language` parameter specifies the language of the search query provided in the `query` parameter for translation into Chinese. This is important because the query must be in Chinese to be searched on the platform.
        
    """
    url = f"https://pinduoduo-Pin-Duo-Duo-data-service1.p.rapidapi.com/search/searchSuggestions"
    querystring = {'query': query, }
    if target_language:
        querystring['target_language'] = target_language
    if query_language:
        querystring['query_language'] = query_language
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "pinduoduo-Pin-Duo-Duo-data-service1.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def search_simple(query: str, sort: int=None, size: int=None, instock: bool=None, target_language: str=None, maxrate: int=None, minprice: int=None, listid: int=None, maxprice: int=None, query_language: str=None, hasdiscount: bool=None, minrate: int=None, page: int=None, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Searches for items using a query string"
    query: Search query
        sort: This parameter is used to specify the sorting order of the items returned in the API response. There are seven available sorting options for the `sort` parameter:

`default`: This option is the default sort order used by Pinduoduo.

`latest`: This option sorts the items based on their date of release, with the newest items appearing first in the response.

`sales_desc`: This option sorts the items in descending order of their sales volume, with the best-selling items appearing first in the response.

`sales_asc`: This option sorts the items in ascending order of their sales volume, with the least-selling items appearing first in the response.

`price_asc`: This option sorts the items in ascending order of their prices, with the lowest-priced items appearing first in the response.

`price_desc`: This option sorts the items in descending order of their prices, with the highest-priced items appearing first in the response.

`commission_ratio_desc`: This option sorts the items in descending order of their commission ratio, with the highest commission ratio items appearing first in the response.

        size: Number of results per page.
        instock: If set to true, the results will contain only items in stock, if set to false the results will contain only items out of stock. Don't set the parameter so you can get both results.
        target_language: The language of translation, list of all supported languages can be found [here](https://rapidapi.com/iamEvara/api/pinduoduo-Pin-Duo-Duo-data-service1/tutorials/list-of-all-supported-languages).
        maxrate: Maximum Item Rating
        minprice: Minimum Item Price
        listid: Search Identifier
        maxprice: Maximum Item Price
        query_language: The `query_language` parameter specifies the language of the search query provided in the `query` parameter for translation into Chinese. This is important because the query must be in Chinese to be searched on the platform.
        hasdiscount: If set to true, the results will contain only items with discount, if set to false the results will contain only items without a discount. Don't set the parameter so you can get both results.
        minrate: Minimum Item Rating
        page: The page number of the results to be retrieved. Default is 1.
        
    """
    url = f"https://pinduoduo-Pin-Duo-Duo-data-service1.p.rapidapi.com/search/searchSimple"
    querystring = {'query': query, }
    if sort:
        querystring['sort'] = sort
    if size:
        querystring['size'] = size
    if instock:
        querystring['inStock'] = instock
    if target_language:
        querystring['target_language'] = target_language
    if maxrate:
        querystring['maxRate'] = maxrate
    if minprice:
        querystring['minPrice'] = minprice
    if listid:
        querystring['listId'] = listid
    if maxprice:
        querystring['maxPrice'] = maxprice
    if query_language:
        querystring['query_language'] = query_language
    if hasdiscount:
        querystring['hasDiscount'] = hasdiscount
    if minrate:
        querystring['minRate'] = minrate
    if page:
        querystring['page'] = page
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "pinduoduo-Pin-Duo-Duo-data-service1.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def coupon_info(couponsign: str, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Retrieves detailed information about a specific coupon on Pinduoduo, returning simple data about the item."
    couponsign: The `couponSign` parameter is used to identify the specific coupon and retrieve information about it. A coupon sign is a unique identifier that is assigned to a coupon associated with a product or item.
        
    """
    url = f"https://pinduoduo-Pin-Duo-Duo-data-service1.p.rapidapi.com/coupon/couponInfo"
    querystring = {'couponSign': couponsign, }
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "pinduoduo-Pin-Duo-Duo-data-service1.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def search_by_image(imageurl: str, sort: str=None, size: int=None, target_language: str=None, page: int=None, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Searches for items based on an uploaded image"
    imageurl: The Url of the image being searched.
        sort: This parameter is used to specify the sorting order of the items returned in the API response. There are six available sorting options for the `sort` parameter:

`default`: This is the default sort option, and it sorts the items based on the Pinduoduo platform's default sort order.

`price_asc`: This option sorts the items in ascending order of their prices, with the lowest priced items appearing first in the response.

`price_desc`: This option sorts the items in descending order of their prices, with the highest priced items appearing first in the response.

`total_price_asc`: This option sorts the items in ascending order of their total prices, which includes any discounts or promotions, with the lowest total priced items appearing first in the response.

`total_price_desc`: This option sorts the items in descending order of their total prices, with the highest total priced items appearing first in the response.

`volume_desc`: This option sorts the items in descending order of their sales volume, with the best-selling items appearing first in the response.

`vendor_rating_desc`: This option sorts the items in descending order of their seller rating, with the highest rated sellers appearing first in the response.

`updated_time_desc`: This option sorts the items in descending order of their update time, with the most recently updated items appearing first in the response.

        size: Number of results per page.
        target_language: The language of translation, list of all supported languages can be found [here](https://rapidapi.com/iamEvara/api/pinduoduo-Pin-Duo-Duo-data-service1/tutorials/list-of-all-supported-languages).
        page: The page number of the results to be retrieved. Default is 1.
        
    """
    url = f"https://pinduoduo-Pin-Duo-Duo-data-service1.p.rapidapi.com/search/searchByImage"
    querystring = {'imageUrl': imageurl, }
    if sort:
        querystring['sort'] = sort
    if size:
        querystring['size'] = size
    if target_language:
        querystring['target_language'] = target_language
    if page:
        querystring['page'] = page
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "pinduoduo-Pin-Duo-Duo-data-service1.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def coupons_recommendations(page: int=None, size: int=None, target_language: str=None, channeltype: int=None, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Returns a list of recommended coupons."
    page: The page number of the results to be retrieved. Default is 1.
        size: Number of results per page.
        target_language: The language of translation, list of all supported languages can be found [here](https://rapidapi.com/iamEvara/api/pinduoduo-Pin-Duo-Duo-data-service1/tutorials/list-of-all-supported-languages).
        channeltype: Channel type, default is 1:
- `0`: '1.9 Free Shipping',
- `1`: 'Today's Hot Items'
- `2`: 'Brand Clearance'

        
    """
    url = f"https://pinduoduo-Pin-Duo-Duo-data-service1.p.rapidapi.com/coupon/couponsRecommendations"
    querystring = {}
    if page:
        querystring['page'] = page
    if size:
        querystring['size'] = size
    if target_language:
        querystring['target_language'] = target_language
    if channeltype:
        querystring['channelType'] = channeltype
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "pinduoduo-Pin-Duo-Duo-data-service1.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

