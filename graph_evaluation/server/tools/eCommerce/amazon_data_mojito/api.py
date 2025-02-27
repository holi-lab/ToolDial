import requests
import json
from datetime import date, datetime, timedelta
import os

from typing import Optional, Dict, Union, List


def get_product_slim_bulk(countrycode: str, asins: str, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Get less detailed product information BUT in Bulk"
    countrycode: CountryCode is a 2 char string.
        asins: ASINs is a comma separated unique string that represents multiple products. Example:'B082FQS4TM,B00WFC8AX1,B00WRC8AU0,B0988SPN9R'
        
    """
    url = f"https://amazon-data-mojito.p.rapidapi.com/GetProductSlims"
    querystring = {'countryCode': countrycode, 'asins': asins, }
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "amazon-data-mojito.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def get_product_slim(asin: str, countrycode: str, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Get less detailed product information"
    asin: ASIN is a unique product ID.
        countrycode: CountryCode is a 2 char string.
        
    """
    url = f"https://amazon-data-mojito.p.rapidapi.com/GetProductSlim"
    querystring = {'asin': asin, 'countryCode': countrycode, }
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "amazon-data-mojito.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def get_product_full(countrycode: str, asin: str, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Get detailed product information"
    countrycode: CountryCode is a 2 char string.
        asin: ASIN is a unique product ID.
        
    """
    url = f"https://amazon-data-mojito.p.rapidapi.com/GetProductFull"
    querystring = {'countryCode': countrycode, 'asin': asin, }
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "amazon-data-mojito.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

