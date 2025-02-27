import requests
import json
from datetime import date, datetime, timedelta
import os

from typing import Optional, Dict, Union, List


def get_amazon_product_offers(productid: str, api_key: str=None, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "You can get the product offers of any Amazon product  that you do a search on.  This will return your product offers in JSON format to make things more readable."
    
    """
    url = f"https://wm-amazon-scraper.p.rapidapi.com/products/{productid}/offers"
    querystring = {}
    if api_key:
        querystring['api_key'] = api_key
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "wm-amazon-scraper.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def get_amazon_product_reviews(api_key: str, productid: str, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "You can get the reviews of any Amazon product  you choose.  This makes things so much easier and will return your product reviews in JSON format."
    
    """
    url = f"https://wm-amazon-scraper.p.rapidapi.com/products/{productid}/reviews"
    querystring = {'api_key': api_key, }
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "wm-amazon-scraper.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def get_amazon_search_results(searchquery: str, api_key: str, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Get the Amazon search results in an easy and structured manner."
    
    """
    url = f"https://wm-amazon-scraper.p.rapidapi.com/search/{searchquery}"
    querystring = {'api_key': api_key, }
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "wm-amazon-scraper.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def get_amazon_product_dedails(api_key: str, productid: str, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "You can get the details of any Amazon product details of your choice.  This will return your product details in JSON format."
    
    """
    url = f"https://wm-amazon-scraper.p.rapidapi.com/products/{productid}"
    querystring = {'api_key': api_key, }
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "wm-amazon-scraper.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

