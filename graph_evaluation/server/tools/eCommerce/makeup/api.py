import requests
import json
from datetime import date, datetime, timedelta
import os

from typing import Optional, Dict, Union, List


def products(price_less_than: str=None, product_type: str=None, product_tags: str=None, brand: str='colourpop', rating_less_than: str=None, product_category: str='lipstick', price_greater_than: str=None, rating_greater_than: str=None, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Search makeup products"
    price_less_than: see API details
        product_type: product type
        product_tags: see API details
        brand: Examples: maybelline, covergirl
        rating_less_than: see API details
        product_category: Sub-category for each makeup-type. (ie. lip gloss is a category of lipstick). See product types below. If a category exists it will be under 'By Category'. Will return a list of all products of this category
        price_greater_than: see API details
        rating_greater_than: see API details
        
    """
    url = f"https://makeup.p.rapidapi.com/products.json"
    querystring = {}
    if price_less_than:
        querystring['price_less_than'] = price_less_than
    if product_type:
        querystring['product_type'] = product_type
    if product_tags:
        querystring['product_tags'] = product_tags
    if brand:
        querystring['brand'] = brand
    if rating_less_than:
        querystring['rating_less_than'] = rating_less_than
    if product_category:
        querystring['product_category'] = product_category
    if price_greater_than:
        querystring['price_greater_than'] = price_greater_than
    if rating_greater_than:
        querystring['rating_greater_than'] = rating_greater_than
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "makeup.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

