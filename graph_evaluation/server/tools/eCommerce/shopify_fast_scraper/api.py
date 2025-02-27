import requests
import json
from datetime import date, datetime, timedelta
import os

from typing import Optional, Dict, Union, List


def get_the_store_s_products(url: str, groupbycollection: bool=None, page: int=1, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Return the Shopify store as a JSON object from the online url of the store. The collections are returned and each collection contains the products linked to the collection. Pages of five collections are returned at a time else the JSON object gets really big. If no page is passed as parameter the first one is retrieved (page 0). Each query will return the current_page, the page_start and page_end.
		
		If a collection has more than 500 products, only the url of the products are returned and not the full products data. For each product url the endpoint /product can then be used with the product url in order to get the full product data."
    groupbycollection: It is true by default, the products are returned per collection. If products are not part of a collection they will not be returned. By setting groupByCollection to false, you will get a page of 30 products, incrementing the page number will get you the other products, once you receive an empty array as products, it means there are no more page available.
        
    """
    url = f"https://shopify-fast-scraper.p.rapidapi.com/store"
    querystring = {'url': url, }
    if groupbycollection:
        querystring['groupByCollection'] = groupbycollection
    if page:
        querystring['page'] = page
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "shopify-fast-scraper.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def get_the_collection_s_products(url: str, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Return the Shopify collection as a JSON object from the online url of the collection. All the products of the collection are also returned.
		
		If a collection has more than 500 products, only the url of the products are returned and not the full products data. For each product url the endpoint /product can then be used with the product url in order to get the full product data."
    
    """
    url = f"https://shopify-fast-scraper.p.rapidapi.com/collection"
    querystring = {'url': url, }
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "shopify-fast-scraper.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def product(url: str, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Return the Shopify product as a JSON object from the online url of the product."
    
    """
    url = f"https://shopify-fast-scraper.p.rapidapi.com/product"
    querystring = {'url': url, }
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "shopify-fast-scraper.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

