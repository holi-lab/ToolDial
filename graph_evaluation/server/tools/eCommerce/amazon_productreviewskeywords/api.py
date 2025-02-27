import requests
import json
from datetime import date, datetime, timedelta
import os

from typing import Optional, Dict, Union, List


def product_search_supports_all_countries(keyword: str, page: int=None, category: str='aps', country: str='US', toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "The endpoint returns products from the Amazon Search Result"
    keyword: Search keyword

For example: **iphone case**
        page: Page number

Default: 1

        category: Perform search in the specific category

To get list of available categories you can use endpoints: ** /categories**

Default is **aps** - means 'All Categories'
        country: Marketplace country

        US: 'www.amazon.com',
        AU: 'www.amazon.com.au',
        BR: 'www.amazon.com.br',
        CA: 'www.amazon.ca',
        CN: 'www.amazon.cn',
        FR: 'www.amazon.fr',
        DE: 'www.amazon.de',
        IN: 'www.amazon.in',
        IT: 'www.amazon.it',
        MX: 'www.amazon.com.mx',
        NL: 'www.amazon.nl',
        SG: 'www.amazon.sg',
        ES: 'www.amazon.es',
        TR: 'www.amazon.com.tr',
        AE: 'www.amazon.ae',
        GB: 'www.amazon.co.uk',
        SE:  'www.amazon.se',
        
    """
    url = f"https://amazon-product-reviews-keywords.p.rapidapi.com/product/search"
    querystring = {'keyword': keyword, }
    if page:
        querystring['page'] = page
    if category:
        querystring['category'] = category
    if country:
        querystring['country'] = country
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "amazon-product-reviews-keywords.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def product_details_supports_all_countries(asin: str, country: str='US', toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "The endpoint returns product details by ASIN"
    asin: Product ASIN

For example: B07XQXZXJC
        country: Marketplace country

        US: 'www.amazon.com',
        AU: 'www.amazon.com.au',
        BR: 'www.amazon.com.br',
        CA: 'www.amazon.ca',
        CN: 'www.amazon.cn',
        FR: 'www.amazon.fr',
        DE: 'www.amazon.de',
        IN: 'www.amazon.in',
        IT: 'www.amazon.it',
        MX: 'www.amazon.com.mx',
        NL: 'www.amazon.nl',
        SG: 'www.amazon.sg',
        ES: 'www.amazon.es',
        TR: 'www.amazon.com.tr',
        AE: 'www.amazon.ae',
        GB: 'www.amazon.co.uk',
        SE:  'www.amazon.se',
        
    """
    url = f"https://amazon-product-reviews-keywords.p.rapidapi.com/product/details"
    querystring = {'asin': asin, }
    if country:
        querystring['country'] = country
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "amazon-product-reviews-keywords.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def product_reviews_supports_all_countries(asin: str, page: int=None, variants: int=1, filter_by_star: str=None, country: str='US', top: str='0', toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "The endpoint returns product reviews"
    asin: Product ASIN

For example: B07XQXZXJC
        page: Page number

Default: 1

1 page can contain 10 reviews
        variants: If a product has multiple product variants, then by default you will receive reviews related to all product variants. If you need reviews only for specified ASIN then set it to **0**

- 1 - reviews for all product variants (default)
- 0 - reviews only for specified product ASIN
        filter_by_star: Filter reviews by stars. Empty value will return all reviews

- '' - return all reviews(default)
- 1 - return only 1 star reviews
- 2 - return only 2 star reviews
- 3 - return only 3 star reviews
- 4 - return only 4 star reviews
- 5 - return only 5 star reviews
        country: Marketplace country

        US: 'www.amazon.com',
        AU: 'www.amazon.com.au',
        BR: 'www.amazon.com.br',
        CA: 'www.amazon.ca',
        CN: 'www.amazon.cn',
        FR: 'www.amazon.fr',
        DE: 'www.amazon.de',
        IN: 'www.amazon.in',
        IT: 'www.amazon.it',
        MX: 'www.amazon.com.mx',
        NL: 'www.amazon.nl',
        SG: 'www.amazon.sg',
        ES: 'www.amazon.es',
        TR: 'www.amazon.com.tr',
        AE: 'www.amazon.ae',
        GB: 'www.amazon.co.uk',
        SE:  'www.amazon.se',
        top: By default reviews will be sorted by \\\"Most Recent\\\" reviews, if you need to sort them by \\\"Top Reviews\\\" then set this value to 1
        
    """
    url = f"https://amazon-product-reviews-keywords.p.rapidapi.com/product/reviews"
    querystring = {'asin': asin, }
    if page:
        querystring['page'] = page
    if variants:
        querystring['variants'] = variants
    if filter_by_star:
        querystring['filter_by_star'] = filter_by_star
    if country:
        querystring['country'] = country
    if top:
        querystring['top'] = top
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "amazon-product-reviews-keywords.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def get_list_of_categories(country: str='US', toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Get list of available categories
		
		In order to search for the product in the specific category you need a valid category name and this endpoint can be used to view available categories by Countries"
    country: Marketplace country

        US: 'www.amazon.com',
        AU: 'www.amazon.com.au',
        BR: 'www.amazon.com.br',
        CA: 'www.amazon.ca',
        CN: 'www.amazon.cn',
        FR: 'www.amazon.fr',
        DE: 'www.amazon.de',
        IN: 'www.amazon.in',
        IT: 'www.amazon.it',
        MX: 'www.amazon.com.mx',
        NL: 'www.amazon.nl',
        SG: 'www.amazon.sg',
        ES: 'www.amazon.es',
        TR: 'www.amazon.com.tr',
        AE: 'www.amazon.ae',
        GB: 'www.amazon.co.uk',
        JP:   'www.amazon.jp',
        
    """
    url = f"https://amazon-product-reviews-keywords.p.rapidapi.com/categories"
    querystring = {}
    if country:
        querystring['country'] = country
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "amazon-product-reviews-keywords.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

