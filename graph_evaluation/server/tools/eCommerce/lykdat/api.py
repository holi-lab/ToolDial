import requests
import json
from datetime import date, datetime, timedelta
import os

from typing import Optional, Dict, Union, List


def text_search(x_api_key: str, query: str, catalog_name: str, order: str=None, brands: str=None, page: int=None, colors: str=None, per_page: int=None, genders: str=None, sort: str=None, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "With the Text Search API, you can search for apparel products from your product catalogs by providing keywords relating to the products you want."
    
    """
    url = f"https://lykdat1.p.rapidapi.com/products/search/text"
    querystring = {'x-api-key': x_api_key, 'query': query, 'catalog_name': catalog_name, }
    if order:
        querystring['order'] = order
    if brands:
        querystring['brands'] = brands
    if page:
        querystring['page'] = page
    if colors:
        querystring['colors'] = colors
    if per_page:
        querystring['per_page'] = per_page
    if genders:
        querystring['genders'] = genders
    if sort:
        querystring['sort'] = sort
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "lykdat1.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

