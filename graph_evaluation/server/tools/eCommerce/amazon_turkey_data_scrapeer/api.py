import requests
import json
from datetime import date, datetime, timedelta
import os

from typing import Optional, Dict, Union, List


def get_product_offers_in_turkey(productid: str, api_key: str, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "This endpoint can get you product offers in amazon tr products. This method is required the product id in the amazon tr page if you don't know you can get search with my method to. And you need to api_key scraperapi you can get easly in this site ; https://www.scraperapi.com/"
    
    """
    url = f"https://amazon-turkey-data-scrapeer.p.rapidapi.com/products/{productid}/offers"
    querystring = {'api_key': api_key, }
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "amazon-turkey-data-scrapeer.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def get_product_reviews_in_turkey(productid: str, api_key: str, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "This endpoint can get you product reviews in amazon tr products. This method is required the product id in the amazon tr page if you don't know you can get search with my method to. And you need to api_key scraperapi you can get easly in this site ; https://www.scraperapi.com/"
    
    """
    url = f"https://amazon-turkey-data-scrapeer.p.rapidapi.com/products/{productid}/reviews"
    querystring = {'api_key': api_key, }
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "amazon-turkey-data-scrapeer.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def search_product_in_turkey(searchquery: str, api_key: str, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "This endpoint can get you search all product in amazon tr . This method is required the search key . And you need to api_key scraperapi you can get easly in this site ; https://www.scraperapi.com/"
    
    """
    url = f"https://amazon-turkey-data-scrapeer.p.rapidapi.com/search/{searchquery}"
    querystring = {'api_key': api_key, }
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "amazon-turkey-data-scrapeer.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def get_product_detail_in_turkey(api_key: str, productid: str, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "This endpoint can get you all product detail in amazon tr product page. This method is required the product id in the amazon tr page if you don't know you can get search with my method to. And you need to api_key scraperapi you can get easly in this site ; https://www.scraperapi.com/"
    
    """
    url = f"https://amazon-turkey-data-scrapeer.p.rapidapi.com/products/{productid}"
    querystring = {'api_key': api_key, }
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "amazon-turkey-data-scrapeer.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

