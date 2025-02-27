import requests
import json
from datetime import date, datetime, timedelta
import os

from typing import Optional, Dict, Union, List


def amazon_product_reviews(url: str=None, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Amazon Product Reviews are feedback left by customers who have purchased and used a product on Amazon's online marketplace. Customers can rate the product on a scale of one to five stars and leave a written review to share their experiences with other potential buyers.
		
		NOTE: REPLACE **B0B3C2R8MP** WITH YOUR OWN PRODUCT ID"
    
    """
    url = f"https://amazon-scraper-api9.p.rapidapi.com/products/B0B3C2R8MP/reviews"
    querystring = {}
    if url:
        querystring['url'] = url
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "amazon-scraper-api9.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def amazon_product_detail(url: str=None, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Amazon Product Detail is a section of an Amazon product listing page that provides comprehensive information about a particular product. This section typically includes a detailed product description, images, price, shipping information, customer reviews, and seller information.
		
		NOTE: REPLACE **B0B3C2R8MP**THIS WITH YOUR OWN AMAZON PRODUCT ID TO GET MORE RESULTS"
    
    """
    url = f"https://amazon-scraper-api9.p.rapidapi.com/products/B0B3C2R8MP"
    querystring = {}
    if url:
        querystring['url'] = url
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "amazon-scraper-api9.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def amazon_product_search_query(url: str=None, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Amazon Product Search Query is the process of searching for products on Amazon's online marketplace using specific keywords or phrases. When a user types a search term into the search bar, Amazon's search engine returns a list of products that match the query. The search results are typically displayed in order of relevance based on a number of factors, including the product's popularity, ratings and reviews, price, and relevance to the search query."
    
    """
    url = f"https://amazon-scraper-api9.p.rapidapi.com/search/ipad"
    querystring = {}
    if url:
        querystring['url'] = url
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "amazon-scraper-api9.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

