import requests
import json
from datetime import date, datetime, timedelta
import os

from typing import Optional, Dict, Union, List


def get_amazon_product_by_search(searchquery: str, api_key: str='your-api_key', toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Get Amazon Product By Search"
    searchquery: Register to  [https://www.scraperapi.com](https://www.scraperapi.com)  and get scraper api_key
        
    """
    url = f"https://ennvaweb-amazon-data-scraper.p.rapidapi.com/search/{searchquery}"
    querystring = {}
    if api_key:
        querystring['api_key'] = api_key
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "ennvaweb-amazon-data-scraper.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def get_amazon_product_reviews(api_key: str, productid: str, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Get Amazon Product Reviews"
    productid: Register to  [https://www.scraperapi.com](https://www.scraperapi.com)  and get scraper api_key
        
    """
    url = f"https://ennvaweb-amazon-data-scraper.p.rapidapi.com/products/{productid}/reviews"
    querystring = {'api_key': api_key, }
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "ennvaweb-amazon-data-scraper.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def get_amazon_product_offers(api_key: str, productid: str, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Get Amazon Product Offers"
    api_key: Register to  [https://www.scraperapi.com](https://www.scraperapi.com)  and get scraper api_key
        
    """
    url = f"https://ennvaweb-amazon-data-scraper.p.rapidapi.com/products/{productid}/offers"
    querystring = {'api_key': api_key, }
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "ennvaweb-amazon-data-scraper.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def get_amazon_product_details(api_key: str, productdid: str, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Get Amazon Product Details"
    api_key: Register to  [https://www.scraperapi.com](https://www.scraperapi.com)  and get scraper api_key
        
    """
    url = f"https://ennvaweb-amazon-data-scraper.p.rapidapi.com/products/{productdid}"
    querystring = {'api_key': api_key, }
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "ennvaweb-amazon-data-scraper.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

