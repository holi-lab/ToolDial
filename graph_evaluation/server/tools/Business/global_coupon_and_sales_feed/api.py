import requests
import json
from datetime import date, datetime, timedelta
import os

from typing import Optional, Dict, Union, List


def store(storeid: str, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Get All Coupons, Sales, Coupons Count, Store Logo, Store URL, Country, About Section... Enter "storeId" to get All the good stuff for a particular store."
    storeid: Enter storeId received from \\\"search endpoint\\\"
        
    """
    url = f"https://global-coupon-and-sales-feed.p.rapidapi.com/store/{storeid}"
    querystring = {}
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "global-coupon-and-sales-feed.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def searchstores(storename: str, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Autocomplete search for stores return StoreId, Name, Coupons Count, Sales, Store Country.... "StoreId" important to get all coupons and sales for a particular store."
    storename: search for stores
        
    """
    url = f"https://global-coupon-and-sales-feed.p.rapidapi.com/search/{storename}"
    querystring = {}
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "global-coupon-and-sales-feed.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def searchstores(storename: str, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Autocomplete search for stores return StoreId, Name, Coupons Count, Sales, Store Country.... "StoreId" important to get all coupons and sales for a particular store."
    storename: search for stores
        
    """
    url = f"https://global-coupon-and-sales-feed.p.rapidapi.com/search/{storename}"
    querystring = {}
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "global-coupon-and-sales-feed.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

