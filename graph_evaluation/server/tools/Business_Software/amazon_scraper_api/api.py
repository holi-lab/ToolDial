import requests
import json
from datetime import date, datetime, timedelta
import os

from typing import Optional, Dict, Union, List


def search_products(api_key: str, searchquery: str, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "An API endpoint that allows searching products by a specific query, returns a list of matching products along with their details, useful for building search functionality in an e-commerce website or for market research."
    
    """
    url = f"https://amazon-scraper-api4.p.rapidapi.com/search/{searchquery}"
    querystring = {'api_key': api_key, }
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "amazon-scraper-api4.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def get_product_offers(productid: str, api_key: str='f2fb2cfe88aa766c6ee91b82ad8c582c', toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "An API endpoint that returns product offers is a specific URL or endpoint that can be accessed using an API key, where it will respond with information about the available offers for a specific product. This endpoint would likely be part of a larger web scraping or data extraction API that allows users to access a wide range of information from the Amazon website. Depending on the implementation, the endpoint may accept parameters such as product ID or ASIN, as well as options to specify the fields or details to be returned in the response. This endpoint can be useful for businesses or researchers who want to compare prices across different sellers and to find the best deal for a product."
    
    """
    url = f"https://amazon-scraper-api4.p.rapidapi.com/products/{productid}/offers"
    querystring = {}
    if api_key:
        querystring['api_key'] = api_key
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "amazon-scraper-api4.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def get_product_reviews(productid: str, api_key: str='f2fb2cfe88aa766c6ee91b82ad8c582c', toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "This API endpoint returns product reviews of a specific URL or endpoint that can be accessed using an API key, where it will respond with customer reviews for a specific product. This information typically includes details such as the review text, rating, and the date of the review. This endpoint would likely be part of a larger web scraping or data extraction API that allows users to access a wide range of information from the Amazon website. Depending on the implementation, the endpoint may accept parameters such as product ID or ASIN, as well as options to specify the fields or details to be returned in the response. This endpoint is useful for businesses or researchers who want to gain insights on the customer perception of a product."
    
    """
    url = f"https://amazon-scraper-api4.p.rapidapi.com/products/{productid}/reviews"
    querystring = {}
    if api_key:
        querystring['api_key'] = api_key
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "amazon-scraper-api4.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def get_product_details(api_key: str, productid: str, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "An API endpoint that returns product details is a specific URL or endpoint that can be accessed using an API key, where it will respond with information about a specific product. This information typically includes details such as the product's name, description, price, and images. This endpoint would likely be part of a larger web scraping or data extraction API that allows users to access a wide range of information from the Amazon website. Depending on the implementation, the endpoint may accept parameters such as product ID or ASIN, as well as options to specify the fields or details to be returned in the response."
    
    """
    url = f"https://amazon-scraper-api4.p.rapidapi.com/products/{productid}"
    querystring = {'api_key': api_key, }
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "amazon-scraper-api4.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

