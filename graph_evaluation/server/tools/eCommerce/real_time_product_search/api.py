import requests
import json
from datetime import date, datetime, timedelta
import os

from typing import Optional, Dict, Union, List


def search(q: str, min_rating: str=None, product_condition: str=None, max_shipping_days: int=None, store_id: str=None, on_sale: bool=None, free_returns: bool=None, free_shipping: bool=None, max_price: int=None, language: str='en', sort_by: str=None, country: str='us', min_price: int=None, page: int=None, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Search for product offers - both free-form queries and GTIN/EAN are supported. Each page contains up to 30 product offers. Infinite pagination/scrolling is supported using the *page* parameter."
    q: Free-form search query or a GTIN/EAN (e.g. *0194252014233*).
        min_rating: Return products with rating greater than the specified value.
Possible values: `1`, `2`, `3`, `4`.
        product_condition: Only return products with a specific condition.
Possible values: `NEW`, `USED`, `REFURBISHED`.
        max_shipping_days: Only return product offers that offer shipping/delivery of up to specific number of days (i.e. shipping speed).
        store_id: Only return product offers from specific stores (comma separated list of store id's). Store IDs can be obtained from the Google Shopping URL after using the **Seller** filter by taking the part after the `merchagg:` variable within the `tbs` parameter.

When filtering for a certain Seller / Store on Google Shopping, a URL similar to the following is shown on the address bar: `https://www.google.com/search?gl=us&tbm=shop&q=shoes&tbs=mr:1,merchagg:m100456214|m114373355`, in that case, the Store IDs are **m100456214** and **m114373355** - to filter for these stores, set store_id=m100456214,m114373355.
        on_sale: Only return product offers that are currently on sale.
Default: `false`.
        free_returns: Only return product offers that offer free returns.
Default: `false`.
        free_shipping: Only return product offers that offer free shipping/delivery.
Default: `false`.
        max_price: Only return product offers with price lower than a certain value.
        language: The language of the results.
Valid values: see https://en.wikipedia.org/wiki/List_of_ISO_639-1_codes
Default: `en`.
        sort_by: Sort product offers by best match, top rated, lowest or highest price.
Possible values: `BEST_MATCH`, `TOP_RATED`, `LOWEST_PRICE`, `HIGHEST_PRICE`.
Default: `BEST_MATCH`.
        country: Country code of the region/country to return offers for.
Valid values: see https://en.wikipedia.org/wiki/ISO_3166-1_alpha-2
Default: `us`.
        min_price: Only return product offers with price greater than a certain value.
        page: Results page to return.
Default: `1`.
        
    """
    url = f"https://real-time-product-search.p.rapidapi.com/search"
    querystring = {'q': q, }
    if min_rating:
        querystring['min_rating'] = min_rating
    if product_condition:
        querystring['product_condition'] = product_condition
    if max_shipping_days:
        querystring['max_shipping_days'] = max_shipping_days
    if store_id:
        querystring['store_id'] = store_id
    if on_sale:
        querystring['on_sale'] = on_sale
    if free_returns:
        querystring['free_returns'] = free_returns
    if free_shipping:
        querystring['free_shipping'] = free_shipping
    if max_price:
        querystring['max_price'] = max_price
    if language:
        querystring['language'] = language
    if sort_by:
        querystring['sort_by'] = sort_by
    if country:
        querystring['country'] = country
    if min_price:
        querystring['min_price'] = min_price
    if page:
        querystring['page'] = page
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "real-time-product-search.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def product_offers(product_id: str, country: str='us', language: str='en', toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Get all offers available for a product."
    product_id: Product id of the product for which to fetch offers.
        country: Country code of the region/country to return offers for.
Valid values: see https://en.wikipedia.org/wiki/ISO_3166-1_alpha-2
Default: `us`.
        language: The language of the results.
Valid values: see https://en.wikipedia.org/wiki/List_of_ISO_639-1_codes
Default: `en`.
        
    """
    url = f"https://real-time-product-search.p.rapidapi.com/product-offers"
    querystring = {'product_id': product_id, }
    if country:
        querystring['country'] = country
    if language:
        querystring['language'] = language
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "real-time-product-search.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def product_reviews(product_id: str, country: str='us', language: str='en', offset: str=None, rating: str=None, limit: str=None, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Get all product reviews. Infinite pagination/scrolling is supported using the *limit* and *offset* parameters."
    product_id: Product id of the product for which to fetch reviews.
        country: Country code of the region/country.
Valid values: see https://en.wikipedia.org/wiki/ISO_3166-1_alpha-2
Default: `us`.
        language: The language of the results.
Valid values: see https://en.wikipedia.org/wiki/List_of_ISO_639-1_codes
Default: `en`.
        offset: Number of reviews to skip.
Valid values: integers from 0-30000
Default: `0`.
        rating: Only return reviews with user rating greater than the specified value.
Valid values: 1-5.
        limit: Maximum number of product reviews to return.
Valid values: integers from 0-100.
Default: `10`.
        
    """
    url = f"https://real-time-product-search.p.rapidapi.com/product-reviews"
    querystring = {'product_id': product_id, }
    if country:
        querystring['country'] = country
    if language:
        querystring['language'] = language
    if offset:
        querystring['offset'] = offset
    if rating:
        querystring['rating'] = rating
    if limit:
        querystring['limit'] = limit
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "real-time-product-search.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def product_details(product_id: str, country: str='us', language: str='en', toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Get the details of a specific product by product id. Returns the full product details in addition to reviews sample, photos, product specs and more information."
    product_id: Product id of the product for which to get full details.
        country: Country code of the region/country to return offers for.
Valid values: see https://en.wikipedia.org/wiki/ISO_3166-1_alpha-2
Default: `us`.
        language: The language of the results.
Valid values: see https://en.wikipedia.org/wiki/List_of_ISO_639-1_codes
Default: `en`.
        
    """
    url = f"https://real-time-product-search.p.rapidapi.com/product-details"
    querystring = {'product_id': product_id, }
    if country:
        querystring['country'] = country
    if language:
        querystring['language'] = language
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "real-time-product-search.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

