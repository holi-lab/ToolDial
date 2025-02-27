import requests
import json
from datetime import date, datetime, timedelta
import os

from typing import Optional, Dict, Union, List


def bulk_data_api(asins: str, marketplace: str, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Get price/title/prime/review/rating of amazon items in bulk, up to 1000 at a time!"
    asins: List of asins, comma-separated. Eg: B07FY1VJ99,B07MH1SZSW,B07GTJPLFZ
        
    """
    url = f"https://amazon-price.p.rapidapi.com/azapi-bulkPrice"
    querystring = {'asins': asins, 'marketplace': marketplace, }
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "amazon-price.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def competitive_offers_api(page: int, asin: str, marketplace: str, condition: str='new', toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Get all competitive offers for a product, up to 10 per request"
    asin: ASIN of the product. Eg: B07FY1VJ99
        condition: Condition (new/all)
        
    """
    url = f"https://amazon-price.p.rapidapi.com/azapi-allOffers"
    querystring = {'page': page, 'asin': asin, 'marketplace': marketplace, }
    if condition:
        querystring['condition'] = condition
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "amazon-price.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def search_api_us_only(prime: str, query: str, page: int, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Get search results from amazon.com!"
    prime: true/false; if true, will show only prime items
        query: query to search with
        page: page number
        
    """
    url = f"https://amazon-price.p.rapidapi.com/azapi-azSearch"
    querystring = {'prime': prime, 'query': query, 'page': page, }
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "amazon-price.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

