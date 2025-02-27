import requests
import json
from datetime import date, datetime, timedelta
import os

from typing import Optional, Dict, Union, List


def recent_coupons(country: str, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Get coupons that got posted recently. You can specify the country that you want the coupons to be applicable to."
    country: Try US, UK, AU...
        
    """
    url = f"https://kuponiko.p.rapidapi.com/recent/{country}"
    querystring = {}
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "kuponiko.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def coupons_from_specific_seller(seller_id: str, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Gives you the coupons from Amazon, Ebay and 40.000 other shops."
    seller_id: Returns the coupons from Amazon
        
    """
    url = f"https://kuponiko.p.rapidapi.com/seller"
    querystring = {'seller_id': seller_id, }
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "kuponiko.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def seller_by_domain(domain: str, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Search the seller based on their website domain"
    domain: Returns the all matching sellers that relate to amazon.com
        
    """
    url = f"https://kuponiko.p.rapidapi.com/id_by_website"
    querystring = {'domain': domain, }
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "kuponiko.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

