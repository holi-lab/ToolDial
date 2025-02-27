import requests
import json
from datetime import date, datetime, timedelta
import os

from typing import Optional, Dict, Union, List


def offers(is_id: str, latitude: str='37.777805', longitude: str='-122.49493', country: str='US', toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "list offers for the product"
    id: An identifier can be any of the following:

- UPC
- EAN
- ISBN
- GTIN
- Amazon ASIN
        
    """
    url = f"https://price-comparison1.p.rapidapi.com/{is_id}/offers"
    querystring = {}
    if latitude:
        querystring['latitude'] = latitude
    if longitude:
        querystring['longitude'] = longitude
    if country:
        querystring['country'] = country
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "price-comparison1.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def details(is_id: str, country: str='US', longitude: str='-122.49493', latitude: str='37.777805', toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "get details on the product"
    id: An identifier can be any of the following:

- UPC
- EAN
- ISBN
- GTIN
- Amazon ASIN
        
    """
    url = f"https://price-comparison1.p.rapidapi.com/{is_id}"
    querystring = {}
    if country:
        querystring['country'] = country
    if longitude:
        querystring['longitude'] = longitude
    if latitude:
        querystring['latitude'] = latitude
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "price-comparison1.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

