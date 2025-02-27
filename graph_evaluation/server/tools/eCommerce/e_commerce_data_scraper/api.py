import requests
import json
from datetime import date, datetime, timedelta
import os

from typing import Optional, Dict, Union, List


def get_product(productid: str, api_key: str, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Get product details completely."
    api_key: Add your unique API key here. 

Follow these few simple steps to get your API key and keep privately:

1. Go to the Scraper website (https://www.scraperapi.com/).
2. Create your account in few clicks. You can also signup using Google or GitHub.
3. Once logged in, you will see your API key on the top. Just copied that and paste it into the field.
        
    """
    url = f"https://e-commerce-data-scraper.p.rapidapi.com/products/{productid}"
    querystring = {'api_key': api_key, }
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "e-commerce-data-scraper.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def get_search_results(searchterm: str, api_key: str, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "This endpoint will send you all the search results with a particular search term. For example, MacBook pro"
    
    """
    url = f"https://e-commerce-data-scraper.p.rapidapi.com/search/{searchterm}"
    querystring = {'api_key': api_key, }
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "e-commerce-data-scraper.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def get_products_reviews(productid: str, api_key: str, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "You will get all the reviews associated with the product."
    api_key: Add your unique API key here.

Follow these few simple steps to get your API key and keep privately:

1. Go to the Scraper website (https://www.scraperapi.com/).
2. Create your account in few clicks. You can also signup using Google or GitHub.
3. Once logged in, you will see your API key on the top. Just copied that and paste it into the field.
        
    """
    url = f"https://e-commerce-data-scraper.p.rapidapi.com/products/{productid}/reviews"
    querystring = {'api_key': api_key, }
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "e-commerce-data-scraper.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def get_product_offers(api_key: str, productid: str, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "You'll all the offers that you can avail on a particular product."
    api_key: Add your unique API key here.

Follow these few simple steps to get your API key and keep privately:

1. Go to the Scraper website (https://www.scraperapi.com/).
2. Create your account in few clicks. You can also signup using Google or GitHub.
3. Once logged in, you will see your API key on the top. Just copied that and paste it into the field.
        
    """
    url = f"https://e-commerce-data-scraper.p.rapidapi.com/products/{productid}/offers"
    querystring = {'api_key': api_key, }
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "e-commerce-data-scraper.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

