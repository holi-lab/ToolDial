import requests
import json
from datetime import date, datetime, timedelta
import os

from typing import Optional, Dict, Union, List


def search_sellers(query: int, query_language: str=None, getitemsinshop: bool=None, page: int=None, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Searches for sellers using a query string"
    query: Search query
        query_language: The `query_language` parameter specifies the language of the search query provided in the `query` parameter for translation into Chinese. This is important because the query must be in Chinese to be searched on the platform.
        getitemsinshop: Determines whether the response should include the full information about the items(one item only) being sold by the seller or just return its ID.
        page: The page number of the results to be retrieved. Default is 1.
        
    """
    url = f"https://taobao-tmall-Tao-Bao-data-service.p.rapidapi.com/seller/searchSellers"
    querystring = {'query': query, }
    if query_language:
        querystring['query_language'] = query_language
    if getitemsinshop:
        querystring['getItemsInShop'] = getitemsinshop
    if page:
        querystring['page'] = page
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "taobao-tmall-Tao-Bao-data-service.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def all_categories(toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Retrieves all categories available"
    
    """
    url = f"https://taobao-tmall-Tao-Bao-data-service.p.rapidapi.com/category/allCategories"
    querystring = {}
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "taobao-tmall-Tao-Bao-data-service.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def search_category_items(query: int, categoryid: int, sort: str=None, minsellerrating: int=None, page: int=None, istmall: bool=None, size: int=None, maxprice: int=None, minprice: int=None, target_language: str=None, query_language: int=None, maxsellerrating: int=None, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Searches for items under a specific category using a query string"
    query: Search query
        categoryid: Category Identifier
        sort: This parameter is used to specify the sorting order of the items returned in the API response. There are six available sorting options for the `sort` parameter:

`default`: This is the default sort option, and it sorts the items based on the Pinduoduo platform's default sort order.

`price_asc`: This option sorts the items in ascending order of their prices, with the lowest priced items appearing first in the response.

`price_desc`: This option sorts the items in descending order of their prices, with the highest priced items appearing first in the response.

`total_price_asc`: This option sorts the items in ascending order of their total prices, which includes any discounts or promotions, with the lowest total priced items appearing first in the response.

`total_price_desc`: This option sorts the items in descending order of their total prices, with the highest total priced items appearing first in the response.

`volume_desc`: This option sorts the items in descending order of their sales volume, with the best-selling items appearing first in the response.

`vendor_rating_desc`: This option sorts the items in descending order of their seller rating, with the highest rated sellers appearing first in the response.

`updated_time_desc`: This option sorts the items in descending order of their update time, with the most recently updated items appearing first in the response.

        minsellerrating: Minimum Item Seller Rating
        page: The page number of the results to be retrieved. Default is 1.
        istmall: Indicates if the search results would include only Tmall products. Set to true if you only want Tmall products, false if you want only Taobao products, don't provide it if you want both.
        size: Number of results per page.
        maxprice: Maximum Item Price
        minprice: Minimum Item Price
        target_language: The language of translation, list of all supported languages can be found [here](https://rapidapi.com/iamEvara/api/taobao-tmall-Tao-Bao-data-service/tutorials/list-of-all-supported-languages).
        query_language: The `query_language` parameter specifies the language of the search query provided in the `query` parameter for translation into Chinese. This is important because the query must be in Chinese to be searched on the platform.
        maxsellerrating: Maximum Item Seller Rating
        
    """
    url = f"https://taobao-tmall-Tao-Bao-data-service.p.rapidapi.com/category/searchCategoryItems"
    querystring = {'query': query, 'categoryId': categoryid, }
    if sort:
        querystring['sort'] = sort
    if minsellerrating:
        querystring['minSellerRating'] = minsellerrating
    if page:
        querystring['page'] = page
    if istmall:
        querystring['isTmall'] = istmall
    if size:
        querystring['size'] = size
    if maxprice:
        querystring['maxPrice'] = maxprice
    if minprice:
        querystring['minPrice'] = minprice
    if target_language:
        querystring['target_language'] = target_language
    if query_language:
        querystring['query_language'] = query_language
    if maxsellerrating:
        querystring['maxSellerRating'] = maxsellerrating
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "taobao-tmall-Tao-Bao-data-service.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def sub_categories(categoryid: int, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Retrieves subcategories for a specific category"
    categoryid: Category Identifier
        
    """
    url = f"https://taobao-tmall-Tao-Bao-data-service.p.rapidapi.com/category/subCategories"
    querystring = {'categoryId': categoryid, }
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "taobao-tmall-Tao-Bao-data-service.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def parent_categories(categoryid: int, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Retrieves parent categories for a specific category"
    categoryid: Category Identifier
        
    """
    url = f"https://taobao-tmall-Tao-Bao-data-service.p.rapidapi.com/category/parentCategories"
    querystring = {'categoryId': categoryid, }
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "taobao-tmall-Tao-Bao-data-service.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def category_items(categoryid: int, sort: str=None, page: int=None, maxsellerrating: int=None, minprice: int=None, minsellerrating: int=None, size: int=None, target_language: str=None, istmall: bool=None, maxprice: int=None, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Retrieves items under a specific category"
    categoryid: Category Identifier
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
        maxsellerrating: Maximum Item Seller Rating
        minprice: Minimum Item Price
        minsellerrating: Minimum Item Seller Rating
        size: Number of results per page.
        target_language: The language of translation, list of all supported languages can be found [here](https://rapidapi.com/iamEvara/api/taobao-tmall-Tao-Bao-data-service/tutorials/list-of-all-supported-languages).
        istmall: Indicates if the search results would include only Tmall products. Set to true if you only want Tmall products, false if you want only Taobao products, don't provide it if you want both.
        maxprice: Maximum Item Price
        
    """
    url = f"https://taobao-tmall-Tao-Bao-data-service.p.rapidapi.com/category/categoryItems"
    querystring = {'categoryId': categoryid, }
    if sort:
        querystring['sort'] = sort
    if page:
        querystring['page'] = page
    if maxsellerrating:
        querystring['maxSellerRating'] = maxsellerrating
    if minprice:
        querystring['minPrice'] = minprice
    if minsellerrating:
        querystring['minSellerRating'] = minsellerrating
    if size:
        querystring['size'] = size
    if target_language:
        querystring['target_language'] = target_language
    if istmall:
        querystring['isTmall'] = istmall
    if maxprice:
        querystring['maxPrice'] = maxprice
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "taobao-tmall-Tao-Bao-data-service.p.rapidapi.com"
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
    sellerid: Taobao Seller Identifier
        
    """
    url = f"https://taobao-tmall-Tao-Bao-data-service.p.rapidapi.com/seller/sellerInfo"
    querystring = {'sellerId': sellerid, }
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "taobao-tmall-Tao-Bao-data-service.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def search_sellers_items(sellerid: str, query: str, maxsellerrating: int=None, minprice: int=None, page: int=None, minsellerrating: int=None, istmall: bool=None, size: int=None, target_language: str=None, maxprice: int=None, sort: str=None, query_language: str=None, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Searches for sellers’ items using a query string"
    sellerid: Taobao Seller Identifier
        query: Search query
        maxsellerrating: Maximum Item Seller Rating
        minprice: Minimum Item Price
        page: The page number of the results to be retrieved. Default is 1.
        minsellerrating: Minimum Item Seller Rating
        istmall: Indicates if the search results would include only Tmall products. Set to true if you only want Tmall products, false if you want only Taobao products, don't provide it if you want both.
        size: Number of results per page.
        target_language: The language of translation, list of all supported languages can be found [here](https://rapidapi.com/iamEvara/api/taobao-tmall-Tao-Bao-data-service/tutorials/list-of-all-supported-languages).
        maxprice: Maximum Item Price
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
        
    """
    url = f"https://taobao-tmall-Tao-Bao-data-service.p.rapidapi.com/seller/searchSellerItems"
    querystring = {'sellerId': sellerid, 'query': query, }
    if maxsellerrating:
        querystring['maxSellerRating'] = maxsellerrating
    if minprice:
        querystring['minPrice'] = minprice
    if page:
        querystring['page'] = page
    if minsellerrating:
        querystring['minSellerRating'] = minsellerrating
    if istmall:
        querystring['isTmall'] = istmall
    if size:
        querystring['size'] = size
    if target_language:
        querystring['target_language'] = target_language
    if maxprice:
        querystring['maxPrice'] = maxprice
    if sort:
        querystring['sort'] = sort
    if query_language:
        querystring['query_language'] = query_language
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "taobao-tmall-Tao-Bao-data-service.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def seller_items(sellerid: str, minsellerrating: int=None, size: int=None, minprice: int=None, istmall: bool=None, target_language: str=None, maxsellerrating: int=None, page: int=None, maxprice: int=None, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Retrieves a list of all seller items"
    sellerid: Taobao Seller Identifier
        minsellerrating: Minimum Item Seller Rating
        size: Number of results per page.
        minprice: Minimum Item Price
        istmall: Indicates if the search results would include only Tmall products. Set to true if you only want Tmall products, false if you want only Taobao products, don't provide it if you want both.
        target_language: The language of translation, list of all supported languages can be found [here](https://rapidapi.com/iamEvara/api/taobao-tmall-Tao-Bao-data-service/tutorials/list-of-all-supported-languages).
        maxsellerrating: Maximum Item Seller Rating
        page: The page number of the results to be retrieved. Default is 1.
        maxprice: Maximum Item Price
        
    """
    url = f"https://taobao-tmall-Tao-Bao-data-service.p.rapidapi.com/seller/sellerItems"
    querystring = {'sellerId': sellerid, }
    if minsellerrating:
        querystring['minSellerRating'] = minsellerrating
    if size:
        querystring['size'] = size
    if minprice:
        querystring['minPrice'] = minprice
    if istmall:
        querystring['isTmall'] = istmall
    if target_language:
        querystring['target_language'] = target_language
    if maxsellerrating:
        querystring['maxSellerRating'] = maxsellerrating
    if page:
        querystring['page'] = page
    if maxprice:
        querystring['maxPrice'] = maxprice
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "taobao-tmall-Tao-Bao-data-service.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def item_video_info(videoid: str, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Retrieves item detailed video information"
    videoid: Video Identifier
        
    """
    url = f"https://taobao-tmall-Tao-Bao-data-service.p.rapidapi.com/item/itemVideoInfo"
    querystring = {'videoId': videoid, }
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "taobao-tmall-Tao-Bao-data-service.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def item_info(itemid: int, target_language: str=None, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Retrieves item information"
    itemid: Taobao Item Identifier
        target_language: The language of translation, list of all supported languages can be found [here](https://rapidapi.com/iamEvara/api/taobao-tmall-Tao-Bao-data-service/tutorials/list-of-all-supported-languages).
        
    """
    url = f"https://taobao-tmall-Tao-Bao-data-service.p.rapidapi.com/item/itemInfo"
    querystring = {'itemId': itemid, }
    if target_language:
        querystring['target_language'] = target_language
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "taobao-tmall-Tao-Bao-data-service.p.rapidapi.com"
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
    itemid: Taobao Item Identifier
        target_language: The language of translation, list of all supported languages can be found [here](https://rapidapi.com/iamEvara/api/taobao-tmall-Tao-Bao-data-service/tutorials/list-of-all-supported-languages).
        
    """
    url = f"https://taobao-tmall-Tao-Bao-data-service.p.rapidapi.com/item/itemFullInfo"
    querystring = {'itemId': itemid, }
    if target_language:
        querystring['target_language'] = target_language
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "taobao-tmall-Tao-Bao-data-service.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def item_info_from_url(url: str, target_language: str=None, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Retrieves item information by a Taobao Url"
    url: Taobao Url
        target_language: The language of translation, list of all supported languages can be found [here](https://rapidapi.com/iamEvara/api/taobao-tmall-Tao-Bao-data-service/tutorials/list-of-all-supported-languages).
        
    """
    url = f"https://taobao-tmall-Tao-Bao-data-service.p.rapidapi.com/item/itemInfoFromUrl"
    querystring = {'url': url, }
    if target_language:
        querystring['target_language'] = target_language
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "taobao-tmall-Tao-Bao-data-service.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def coupon_info(couponid: str, target_language: str=None, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Retrieves coupon information"
    couponid: Coupon Identifier
        target_language: The language of translation, list of all supported languages can be found [here](https://rapidapi.com/iamEvara/api/taobao-tmall-Tao-Bao-data-service/tutorials/list-of-all-supported-languages).
        
    """
    url = f"https://taobao-tmall-Tao-Bao-data-service.p.rapidapi.com/coupon/couponInfo"
    querystring = {'couponId': couponid, }
    if target_language:
        querystring['target_language'] = target_language
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "taobao-tmall-Tao-Bao-data-service.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def item_app_info(itemid: int, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Retrieves original item information from Taobao App"
    itemid: Taobao Item Identifier
        
    """
    url = f"https://taobao-tmall-Tao-Bao-data-service.p.rapidapi.com/item/itemAppInfo"
    querystring = {'itemId': itemid, }
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "taobao-tmall-Tao-Bao-data-service.p.rapidapi.com"
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
    itemids: Specify the IDs of the items for which simple information will be retrieved. The IDs should be provided as a comma-separated list of values (eg. `354583020393,354168059048`).
        target_language: The language of translation, list of all supported languages can be found [here](https://rapidapi.com/iamEvara/api/taobao-tmall-Tao-Bao-data-service/tutorials/list-of-all-supported-languages).
        
    """
    url = f"https://taobao-tmall-Tao-Bao-data-service.p.rapidapi.com/item/itemsBatchInfo"
    querystring = {'itemIds': itemids, }
    if target_language:
        querystring['target_language'] = target_language
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "taobao-tmall-Tao-Bao-data-service.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def item_sku_details(itemid: str, skuid: str, target_language: str=None, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Retrieves item SKU details"
    itemid: Taobao Item Identifier
        skuid: Sku Identifier
        target_language: The language of translation, list of all supported languages can be found [here](https://rapidapi.com/iamEvara/api/taobao-tmall-Tao-Bao-data-service/tutorials/list-of-all-supported-languages).
        
    """
    url = f"https://taobao-tmall-Tao-Bao-data-service.p.rapidapi.com/item/itemSkuDetails"
    querystring = {'itemId': itemid, 'skuId': skuid, }
    if target_language:
        querystring['target_language'] = target_language
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "taobao-tmall-Tao-Bao-data-service.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def search_categories(query: int, query_language: int=None, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Searches for categories using a query string"
    query: Search query
        query_language: The `query_language` parameter specifies the language of the search query provided in the `query` parameter for translation into Chinese. This is important because the query must be in Chinese to be searched on the platform.
        
    """
    url = f"https://taobao-tmall-Tao-Bao-data-service.p.rapidapi.com/category/searchCategories"
    querystring = {'query': query, }
    if query_language:
        querystring['query_language'] = query_language
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "taobao-tmall-Tao-Bao-data-service.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def item_recommendations(itemid: int, target_language: str=None, page: int=None, size: int=None, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Retrieves list of items related to an item"
    itemid: Taobao Item Identifier
        target_language: The language of translation, list of all supported languages can be found [here](https://rapidapi.com/iamEvara/api/taobao-tmall-Tao-Bao-data-service/tutorials/list-of-all-supported-languages).
        page: The page number of the results to be retrieved. Default is 1.
        size: Number of results per page.
        
    """
    url = f"https://taobao-tmall-Tao-Bao-data-service.p.rapidapi.com/item/itemRecommendations"
    querystring = {'itemId': itemid, }
    if target_language:
        querystring['target_language'] = target_language
    if page:
        querystring['page'] = page
    if size:
        querystring['size'] = size
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "taobao-tmall-Tao-Bao-data-service.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def category_info(categoryid: int, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Retrieves information about a specific category"
    categoryid: Category Identifier
        
    """
    url = f"https://taobao-tmall-Tao-Bao-data-service.p.rapidapi.com/category/categoryInfo"
    querystring = {'categoryId': categoryid, }
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "taobao-tmall-Tao-Bao-data-service.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def coupons_orders(end_time: int, start_time: int, size: int=None, page: int=None, target_language: str=None, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Retrieves coupon information"
    end_time: Ending time range for which to retrieve the orders information. The time should be provided as a timestamp.
        start_time: Starting time range for which to retrieve the orders information. The time should be provided as a timestamp.
        size: Number of results per page.
        page: The page number of the results to be retrieved. Default is 1.
        target_language: The language of translation, list of all supported languages can be found [here](https://rapidapi.com/iamEvara/api/taobao-tmall-Tao-Bao-data-service/tutorials/list-of-all-supported-languages).
        
    """
    url = f"https://taobao-tmall-Tao-Bao-data-service.p.rapidapi.com/coupon/couponsOrders"
    querystring = {'end_time': end_time, 'start_time': start_time, }
    if size:
        querystring['size'] = size
    if page:
        querystring['page'] = page
    if target_language:
        querystring['target_language'] = target_language
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "taobao-tmall-Tao-Bao-data-service.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def item_description(itemid: int, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Retrieves item Description"
    itemid: Taobao Item Identifier
        
    """
    url = f"https://taobao-tmall-Tao-Bao-data-service.p.rapidapi.com/item/itemDescription"
    querystring = {'itemId': itemid, }
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "taobao-tmall-Tao-Bao-data-service.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def search_items(query: str, maxprice: int=None, target_language: str=None, size: int=None, instock: bool=None, query_language: str=None, hasdiscount: bool=None, minprice: int=None, minsellerrating: int=None, page: int=None, istmall: bool=None, maxsellerrating: int=None, sort: str=None, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Searches for items using a query string"
    query: Search query
        maxprice: Maximum Item Price
        target_language: The language of translation, list of all supported languages can be found [here](https://rapidapi.com/iamEvara/api/taobao-tmall-Tao-Bao-data-service/tutorials/list-of-all-supported-languages).
        size: Number of results per page.
        instock: If set to true, the results will contain only items in stock, if set to false the results will contain only items out of stock. Don't set the parameter so you can get both results.
        query_language: The `query_language` parameter specifies the language of the search query provided in the `query` parameter for translation into Chinese. This is important because the query must be in Chinese to be searched on the platform.
        hasdiscount: If set to true, the results will contain only items with discount, if set to false the results will contain only items without a discount. Don't set the parameter so you can get both results.
        minprice: Minimum Item Price
        minsellerrating: Minimum Item Seller Rating
        page: The page number of the results to be retrieved. Default is 1.
        istmall: Indicates if the search results would include only Tmall products. Set to true if you only want Tmall products, false if you want only Taobao products, don't provide it if you want both.
        maxsellerrating: Maximum Item Seller Rating
        sort: This parameter is used to specify the sorting order of the items returned in the API response. There are six available sorting options for the `sort` parameter:

`default`: This is the default sort option, and it sorts the items based on the Pinduoduo platform's default sort order.

`price_asc`: This option sorts the items in ascending order of their prices, with the lowest priced items appearing first in the response.

`price_desc`: This option sorts the items in descending order of their prices, with the highest priced items appearing first in the response.

`total_price_asc`: This option sorts the items in ascending order of their total prices, which includes any discounts or promotions, with the lowest total priced items appearing first in the response.

`total_price_desc`: This option sorts the items in descending order of their total prices, with the highest total priced items appearing first in the response.

`volume_desc`: This option sorts the items in descending order of their sales volume, with the best-selling items appearing first in the response.

`vendor_rating_desc`: This option sorts the items in descending order of their seller rating, with the highest rated sellers appearing first in the response.

`updated_time_desc`: This option sorts the items in descending order of their update time, with the most recently updated items appearing first in the response.

        
    """
    url = f"https://taobao-tmall-Tao-Bao-data-service.p.rapidapi.com/search/searchItems"
    querystring = {'query': query, }
    if maxprice:
        querystring['maxPrice'] = maxprice
    if target_language:
        querystring['target_language'] = target_language
    if size:
        querystring['size'] = size
    if instock:
        querystring['inStock'] = instock
    if query_language:
        querystring['query_language'] = query_language
    if hasdiscount:
        querystring['hasDiscount'] = hasdiscount
    if minprice:
        querystring['minPrice'] = minprice
    if minsellerrating:
        querystring['minSellerRating'] = minsellerrating
    if page:
        querystring['page'] = page
    if istmall:
        querystring['isTmall'] = istmall
    if maxsellerrating:
        querystring['maxSellerRating'] = maxsellerrating
    if sort:
        querystring['sort'] = sort
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "taobao-tmall-Tao-Bao-data-service.p.rapidapi.com"
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
        target_language: The language of translation, list of all supported languages can be found [here](https://rapidapi.com/iamEvara/api/taobao-tmall-Tao-Bao-data-service/tutorials/list-of-all-supported-languages).
        
    """
    url = f"https://taobao-tmall-Tao-Bao-data-service.p.rapidapi.com/search/searchSuggestions"
    querystring = {'query': query, }
    if query_language:
        querystring['query_language'] = query_language
    if target_language:
        querystring['target_language'] = target_language
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "taobao-tmall-Tao-Bao-data-service.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def search_by_image(imageurl: str, sort: str=None, maxprice: int=None, page: int=None, size: int=None, istmall: bool=None, minprice: int=None, maxsellerrating: int=None, target_language: str=None, minsellerrating: int=None, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
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

        maxprice: Maximum Item Price
        page: The page number of the results to be retrieved. Default is 1.
        size: Number of results per page.
        istmall: Indicates if the search results would include only Tmall products. Set to true if you only want Tmall products, false if you want only Taobao products, don't provide it if you want both.
        minprice: Minimum Item Price
        maxsellerrating: Maximum Item Seller Rating
        target_language: The language of translation, list of all supported languages can be found [here](https://rapidapi.com/iamEvara/api/taobao-tmall-Tao-Bao-data-service/tutorials/list-of-all-supported-languages).
        minsellerrating: Minimum Item Seller Rating
        
    """
    url = f"https://taobao-tmall-Tao-Bao-data-service.p.rapidapi.com/search/searchByImage"
    querystring = {'imageUrl': imageurl, }
    if sort:
        querystring['sort'] = sort
    if maxprice:
        querystring['maxPrice'] = maxprice
    if page:
        querystring['page'] = page
    if size:
        querystring['size'] = size
    if istmall:
        querystring['isTmall'] = istmall
    if minprice:
        querystring['minPrice'] = minprice
    if maxsellerrating:
        querystring['maxSellerRating'] = maxsellerrating
    if target_language:
        querystring['target_language'] = target_language
    if minsellerrating:
        querystring['minSellerRating'] = minsellerrating
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "taobao-tmall-Tao-Bao-data-service.p.rapidapi.com"
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
    itemid: Taobao Item Identifier
        size: Number of results per page.
        page: The page number of the results to be retrieved. Default is 1.
        target_language: The language of translation, list of all supported languages can be found [here](https://rapidapi.com/iamEvara/api/taobao-tmall-Tao-Bao-data-service/tutorials/list-of-all-supported-languages).
        
    """
    url = f"https://taobao-tmall-Tao-Bao-data-service.p.rapidapi.com/item/itemReviews"
    querystring = {'itemId': itemid, }
    if size:
        querystring['size'] = size
    if page:
        querystring['page'] = page
    if target_language:
        querystring['target_language'] = target_language
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "taobao-tmall-Tao-Bao-data-service.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def item_reviews_detailed(itemid: int, target_language: str=None, size: int=None, page: int=None, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Retrieves all item reviews information, but more detailed"
    itemid: Taobao Item Identifier
        target_language: The language of translation, list of all supported languages can be found [here](https://rapidapi.com/iamEvara/api/taobao-tmall-Tao-Bao-data-service/tutorials/list-of-all-supported-languages).
        size: Number of results per page.
        page: The page number of the results to be retrieved. Default is 1.
        
    """
    url = f"https://taobao-tmall-Tao-Bao-data-service.p.rapidapi.com/item/itemReviewsDetailed"
    querystring = {'itemId': itemid, }
    if target_language:
        querystring['target_language'] = target_language
    if size:
        querystring['size'] = size
    if page:
        querystring['page'] = page
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "taobao-tmall-Tao-Bao-data-service.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

