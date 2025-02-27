import requests
import json
from datetime import date, datetime, timedelta
import os

from typing import Optional, Dict, Union, List


def requests_details_of_a_specific_product(is_id: str, country: str='australia', toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Requests details of a specific product
		
		**Allowed *country* params**
		- global
		- australia
		- austria
		- canada
		- france
		- germany
		- hong kong
		- ireland
		- italy
		- malaysia
		- netherlands
		- philippines
		- poland
		- singapore
		- spain
		- switzerland
		- united kingdom"
    
    """
    url = f"https://ebay-data-scraper.p.rapidapi.com/products/{is_id}"
    querystring = {}
    if country:
        querystring['country'] = country
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "ebay-data-scraper.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def search_products(page_number: int, product_name: str, country: str='canada', toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "This endpoint can be used to search a products based in the name. Another feature is the possibility to navigate between the pages of results.
		You must provider in each request two parameters: page_number and product_name.
		
		**Allowed *country* params**
		- global
		- australia
		- austria
		- canada
		- france
		- germany
		- hong kong
		- ireland
		- italy
		- malaysia
		- netherlands
		- philippines
		- poland
		- singapore
		- spain
		- switzerland
		- united kingdom"
    
    """
    url = f"https://ebay-data-scraper.p.rapidapi.com/products"
    querystring = {'page_number': page_number, 'product_name': product_name, }
    if country:
        querystring['country'] = country
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "ebay-data-scraper.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def request_reviews_by_product_id(is_id: str, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "This endpoint request a specific product's reviews and informations about the seller, for example positive feedback and sold items."
    
    """
    url = f"https://ebay-data-scraper.p.rapidapi.com/products/{is_id}/reviews"
    querystring = {}
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "ebay-data-scraper.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def request_api_status(toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Request API status"
    
    """
    url = f"https://ebay-data-scraper.p.rapidapi.com/status/api"
    querystring = {}
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "ebay-data-scraper.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def request_the_daily_global_home_deals(toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Request the daily global home deals"
    
    """
    url = f"https://ebay-data-scraper.p.rapidapi.com/deals/home"
    querystring = {}
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "ebay-data-scraper.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def request_the_daily_global_fashion_deals(toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Request the daily global fashion deals"
    
    """
    url = f"https://ebay-data-scraper.p.rapidapi.com/deals/fashion"
    querystring = {}
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "ebay-data-scraper.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def request_the_daily_global_tech_deals(toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Request the daily global tech deals"
    
    """
    url = f"https://ebay-data-scraper.p.rapidapi.com/deals/tech"
    querystring = {}
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "ebay-data-scraper.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def request_server_status(toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Without params"
    
    """
    url = f"https://ebay-data-scraper.p.rapidapi.com/status/server"
    querystring = {}
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "ebay-data-scraper.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def request_the_daily_global_featured_deals(toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Request the daily global featured deals"
    
    """
    url = f"https://ebay-data-scraper.p.rapidapi.com/deals"
    querystring = {}
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "ebay-data-scraper.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

