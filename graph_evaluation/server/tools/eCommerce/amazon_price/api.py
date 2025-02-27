import requests
import json
from datetime import date, datetime, timedelta
import os

from typing import Optional, Dict, Union, List


def search(keywords: str, marketplace: str, page: int=0, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Your GET endpoint"
    marketplace: Amazon Marketplace Id
        
    """
    url = f"https://amazon-price1.p.rapidapi.com/search"
    querystring = {'keywords': keywords, 'marketplace': marketplace, }
    if page:
        querystring['page'] = page
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "amazon-price1.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def pricereport(asin: str, marketplace: str, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Your GET endpoint"
    marketplace: Amazon Marketplace Id
        
    """
    url = f"https://amazon-price1.p.rapidapi.com/priceReport"
    querystring = {'asin': asin, 'marketplace': marketplace, }
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "amazon-price1.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def eantoasin(marketplace: str, ean: str, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Your GET endpoint"
    marketplace: Amazon Marketplace Id
        ean: European Article Number
        
    """
    url = f"https://amazon-price1.p.rapidapi.com/eanToAsin"
    querystring = {'marketplace': marketplace, 'ean': ean, }
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "amazon-price1.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def upctoasin(upc: str, marketplace: str, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Your GET endpoint"
    upc: Universal Product Code
        
    """
    url = f"https://amazon-price1.p.rapidapi.com/upcToAsin"
    querystring = {'upc': upc, 'marketplace': marketplace, }
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "amazon-price1.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

