import requests
import json
from datetime import date, datetime, timedelta
import os

from typing import Optional, Dict, Union, List


def search(query: str, brand: str=None, min_price: int=None, max_price: int=None, country: str='US', category_id: str='aps', sort_by: str=None, page: str='1', toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Search for product offers on Amazon with support for multiple filters and options."
    query: Search query. Supports both free-form text queries or a product asin.
        brand: Find products with a specific brand. Multiple brands can be specified as a comma (,) separated list. The brand values can be seen from Amazon's search left filters panel, as seen [here](https://www.amazon.com/s?k=phone).

**e.g.** `SAMSUNG`
**e.g.** `Google,Apple`
        min_price: Only return product offers with price greater than a certain value. Specified in the currency of the selected country. For example, in case country=US, a value of *105.34* means *$105.34*.
        max_price: Only return product offers with price lower than a certain value. Specified in the currency of the selected country. For example, in case country=US, a value of *105.34* means *$105.34*.
        country: Sets the marketplace country, language and currency. 

**Default:** `US`

**Allowed values:**  `US, AU, BR, CA, CN, FR, DE, IN, IT, MX, NL, SG, ES, TR, AE, GB, JP`

        category_id: Find products in a specific category / department. Use the **Product Category List** endpoint to get a list of valid categories and their ids for the country specified in the request.

**Default:** `aps` (All Departments)
        sort_by: Return the results in a specific sort order.

**Default:** `RELEVANCE`

**Allowed values:** `RELEVANCE, LOWEST_PRICE, HIGHEST_PRICE, REVIEWS, NEWEST`

        page: Results page to return.

**Default:** `1`
        
    """
    url = f"https://real-time-amazon-data.p.rapidapi.com/search"
    querystring = {'query': query, }
    if brand:
        querystring['brand'] = brand
    if min_price:
        querystring['min_price'] = min_price
    if max_price:
        querystring['max_price'] = max_price
    if country:
        querystring['country'] = country
    if category_id:
        querystring['category_id'] = category_id
    if sort_by:
        querystring['sort_by'] = sort_by
    if page:
        querystring['page'] = page
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "real-time-amazon-data.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def product_details(asin: str, country: str='US', toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Get additional product information / details such as description, about, rating distribution and specs."
    asin: Product ASIN for which to get details. Supports batching of up to 10 ASINs in a single request, separated by comma (e.g. *B08PPDJWC8,B07ZPKBL9V, B08BHXG144*).

Note that each ASIN in a batch request is counted as a single request against the plan quota.
        country: Sets the marketplace country, language and currency. 

**Default:** `US`

**Allowed values:**  `US, AU, BR, CA, CN, FR, DE, IN, IT, MX, NL, SG, ES, TR, AE, GB, JP`

        
    """
    url = f"https://real-time-amazon-data.p.rapidapi.com/product-details"
    querystring = {'asin': asin, }
    if country:
        querystring['country'] = country
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "real-time-amazon-data.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def product_reviews(asin: str, query: str=None, sort_by: str=None, verified_purchases_only: bool=None, page_size: int=10, page: int=1, star_rating: str=None, images_or_videos_only: bool=None, country: str='US', toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Get and paginate through all product reviews on Amazon."
    asin: Product asin for which to get reviews.
        query: Find reviews matching a search query.
        sort_by: Return reviews in a specific sort order.

**Default:** `TOP_REVIEWS`

**Allowed values:** `TOP_REVIEWS, MOST_RECENT`

        verified_purchases_only: Only return reviews by reviewers who made a verified purchase.
        page_size: Results page size.

**Allowed values:** `1-20`

**Default:** `10`
        page: Results page to return.

**Default:** `1`
        star_rating: Only return reviews with a specific star rating.

**Default:** `ALL`

**Allowed values:** `ALL, 5_STARS, 4_STARS, 3_STARS, 2_STARS, 1_STARS, POSITIVE, CRITICAL`

        images_or_videos_only: Only return reviews containing images and / or videos.
        country: Sets the marketplace country, language and currency. 

**Default:** `US`

**Allowed values:**  `US, AU, BR, CA, CN, FR, DE, IN, IT, MX, NL, SG, ES, TR, AE, GB, JP`

        
    """
    url = f"https://real-time-amazon-data.p.rapidapi.com/product-reviews"
    querystring = {'asin': asin, }
    if query:
        querystring['query'] = query
    if sort_by:
        querystring['sort_by'] = sort_by
    if verified_purchases_only:
        querystring['verified_purchases_only'] = verified_purchases_only
    if page_size:
        querystring['page_size'] = page_size
    if page:
        querystring['page'] = page
    if star_rating:
        querystring['star_rating'] = star_rating
    if images_or_videos_only:
        querystring['images_or_videos_only'] = images_or_videos_only
    if country:
        querystring['country'] = country
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "real-time-amazon-data.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def product_category_list(country: str='US', toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Get Amazon product categories (per country / marketplace)."
    country: Sets the marketplace country, language and currency. 

**Default:** `US`

**Allowed values:**  `US, AU, BR, CA, CN, FR, DE, IN, IT, MX, NL, SG, ES, TR, AE, GB, JP`

        
    """
    url = f"https://real-time-amazon-data.p.rapidapi.com/product-category-list"
    querystring = {}
    if country:
        querystring['country'] = country
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "real-time-amazon-data.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def product_offers(asin: str, delivery: str=None, limit: int=100, product_condition: str=None, country: str='US', toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Get top 10 offers of a specific product on Amazon by its asin, the first offer in the list is the pinned offer returned by the **Search** endpoint. Supports filtering by product condition."
    asin: Product ASIN for which to get offers. Supports batching of up to 10 ASINs in a single request, separated by comma (e.g. *B08PPDJWC8,B07ZPKBL9V, B08BHXG144*).

Note that each ASIN in a batch request is counted as a single request against the plan quota.
        delivery: [EXPERIMENTAL]
Find products with specific delivery option, specified as a comma delimited list of the following values: `PRIME_ELIGIBLE,FREE_DELIVERY`.

**e.g.** `FREE_DELIVERY`
**e.g.** `PRIME_ELIGIBLE,FREE_DELIVERY`

        limit: Maximum number of offers to return.

**Default:** `100`
        product_condition: Find products in specific conditions, specified as a comma delimited list of the following values: `NEW, USED_LIKE_NEW, USED_VERY_GOOD, USED_GOOD, USED_ACCEPTABLE`.

**e.g.** `NEW,USED_LIKE_NEW`
**e.g.** `USED_VERY_GOOD,USED_GOOD,USED_LIKE_NEW`

        country: Sets the marketplace country, language and currency. 

**Default:** `US`

**Allowed values:**  `US, AU, BR, CA, CN, FR, DE, IN, IT, MX, NL, SG, ES, TR, AE, GB, JP`

        
    """
    url = f"https://real-time-amazon-data.p.rapidapi.com/product-offers"
    querystring = {'asin': asin, }
    if delivery:
        querystring['delivery'] = delivery
    if limit:
        querystring['limit'] = limit
    if product_condition:
        querystring['product_condition'] = product_condition
    if country:
        querystring['country'] = country
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "real-time-amazon-data.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

