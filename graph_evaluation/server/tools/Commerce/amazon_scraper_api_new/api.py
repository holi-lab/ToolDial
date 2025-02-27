import requests
import json
from datetime import date, datetime, timedelta
import os

from typing import Optional, Dict, Union, List


def get_amazon_product_reviews(api_key: str, productid: str, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Ever since Amazon removed incentivized reviews and introduced verified and non-verified reviews, how Amazon calculates a product's rating has become increasingly complex. All ratings and reviews are not weighted equally.
		Ecommerce underwent an incredible surge in 2020 as consumers shifted from shopping in-store to online. In 2022, online shopping is a lifestyle mainstay: 48% of consumers shop on Amazon at least once per week, and 61% of consumers say they’re influenced by the product with the best ratings and reviews.
		Product reviews can make or break a new Amazon FBA seller. According to a 2017 study performed by G2 and Heinz Marketing, 92% of customers are more likely to purchase a product after reading a positive review.
		So let's get Our Product Reviews"
    
    """
    url = f"https://amazon-scraper-api-new.p.rapidapi.com/products/{productid}/reviews"
    querystring = {'api_key': api_key, }
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "amazon-scraper-api-new.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def get_amazon_product_offers(api_key: str, productid: str, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Get a  listing of all offers for a product using the API"
    
    """
    url = f"https://amazon-scraper-api-new.p.rapidapi.com/products/{productid}/offers"
    querystring = {'api_key': api_key, }
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "amazon-scraper-api-new.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def get_amazon_search_results(searchquery: str, api_key: str='11a8d8c8d681a5d9902659a72db4a590', toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "The SearchItems operation searches for items on Amazon based on a search query. The Amazon Product Advertising API returns up to ten items per search request.
		
		A SearchItems request requires a search category, which, if not specified, defaults to "All" and value for at least one of the Keywords, Actor, Artist, Author, Brand, or Title for searching items on Amazon.
		
		 However, note that it is mandatory to provide at least one of the above-mentioned parameters."
    
    """
    url = f"https://amazon-scraper-api-new.p.rapidapi.com/search/{searchquery}"
    querystring = {}
    if api_key:
        querystring['api_key'] = api_key
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "amazon-scraper-api-new.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def get_amazon_product_details(api_key: str, productid: str, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "A product description describes your product, what it is and why it’s worth buying.
		It also plays a vital role in Amazon search rankings and conversions.
		Estimates show that there are over 1,750,000 sellers who have products listed on Amazon, each vying for customers’ attention.
		Concise, friendly, and honest descriptions of your products will bring customers back, time and time again"
    
    """
    url = f"https://amazon-scraper-api-new.p.rapidapi.com/products/{productid}"
    querystring = {'api_key': api_key, }
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "amazon-scraper-api-new.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

