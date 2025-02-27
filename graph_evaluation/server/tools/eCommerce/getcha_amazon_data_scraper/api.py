import requests
import json
from datetime import date, datetime, timedelta
import os

from typing import Optional, Dict, Union, List


def get_search_results_for_a_keyword(searchquery: str, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "This endpoint fetches all the results from Amazon.com for a keyword. It also shows the sponsored products."
    
    """
    url = f"https://getcha-amazon-data-scraper.p.rapidapi.com/search/{searchquery}"
    querystring = {}
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "getcha-amazon-data-scraper.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def get_product_details(productid: str, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Writing "/products/(ASIN number of product)" in the URL fetches the same data if searched on Amazon but in JSON format. The API fetches every nook and cranny of data and displays on the screen"
    
    """
    url = f"https://getcha-amazon-data-scraper.p.rapidapi.com/products/{productid}"
    querystring = {}
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "getcha-amazon-data-scraper.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def get_detailed_reviews_for_the_product(productid: str, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "This endpoint is made because while fetching the product details the reviews are not fetched because most of the listings have more than 5000 reviews, and fetching them all will be laggy and too slow. The API only shows top critical and positiTherefore, just adding /reviews in the link at last of the URL in the previous URL gives every single review of the product"
    
    """
    url = f"https://getcha-amazon-data-scraper.p.rapidapi.com/products/{productid}/reviews"
    querystring = {}
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "getcha-amazon-data-scraper.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def get_product_offers(productid: str, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "This endpoint is not that important. It gives all the current offers of a product. Note that most of the products don't have offers"
    
    """
    url = f"https://getcha-amazon-data-scraper.p.rapidapi.com/products/{productid}/offers"
    querystring = {}
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "getcha-amazon-data-scraper.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

