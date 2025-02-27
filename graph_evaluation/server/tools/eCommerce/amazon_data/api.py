import requests
import json
from datetime import date, datetime, timedelta
import os

from typing import Optional, Dict, Union, List


def lookup_product_by_asin_code(asin: str, region: str='us', merchant: str=None, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Get product information by Asin Code and Merchant"
    asin: ASIN code
        region: Default : us ( Region list : us, jp, de, es, in) 
        merchant: Merchant ID
        
    """
    url = f"https://amazon-data.p.rapidapi.com/asin.php"
    querystring = {'asin': asin, }
    if region:
        querystring['region'] = region
    if merchant:
        querystring['merchant'] = merchant
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "amazon-data.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def get_deals_and_offers(page: int=1, region: str='us', toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Get Deals and Offers"
    region: Default : us ( Region list : us, jp, de, es, in )
        
    """
    url = f"https://amazon-data.p.rapidapi.com/deal.php"
    querystring = {}
    if page:
        querystring['page'] = page
    if region:
        querystring['region'] = region
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "amazon-data.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def search_products_by_keywords(keyword: str, region: str='us', page: int=1, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Searching products on amazon by keyword or asin code"
    region: Default : us ( Region list : us, jp, de, es, in )
        
    """
    url = f"https://amazon-data.p.rapidapi.com/search.php"
    querystring = {'keyword': keyword, }
    if region:
        querystring['region'] = region
    if page:
        querystring['page'] = page
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "amazon-data.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def upc_to_asin(upc: str, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Search ASIN by UPC Code"
    
    """
    url = f"https://amazon-data.p.rapidapi.com/upc.php"
    querystring = {'upc': upc, }
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "amazon-data.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

