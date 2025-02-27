import requests
import json
from datetime import date, datetime, timedelta
import os

from typing import Optional, Dict, Union, List


def paidaccess(toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "This endpoint provides access to the full suite of features available to premium or paid users. It allows you to retrieve data with fewer restrictions and higher limits than the free access endpoint."
    
    """
    url = f"https://ai-powered-wishscraper-api1.p.rapidapi.com/wish_products_paid"
    querystring = {}
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "ai-powered-wishscraper-api1.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def freeaccess(toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "This endpoint provides access to the basic features available to free or unpaid users. It allows you to retrieve data according to the limitations and restrictions set for free users."
    
    """
    url = f"https://ai-powered-wishscraper-api1.p.rapidapi.com/wish_products_unpaid"
    querystring = {}
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "ai-powered-wishscraper-api1.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

