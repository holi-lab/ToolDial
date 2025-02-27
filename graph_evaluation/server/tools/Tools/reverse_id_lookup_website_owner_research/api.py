import requests
import json
from datetime import date, datetime, timedelta
import os

from typing import Optional, Dict, Union, List


def analyze(platform: str, value: str, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Reverse lookup ID"
    platform: Which platform to search for. Options: domain, adsense, googleanalytics, amazon_affiliate, email, addthis, sharethis, clickbank_affiliate, commissionjunction_affiliate, doubleclick, linkshare_affiliate, facebook_app, ip, nameserver, registrar, amazon_product, clickbank_product, commissionjunction_product, linkshare_product.
        value: The ID to search from the specific platform.
        
    """
    url = f"https://dait-reverse-website-lookup-v1.p.rapidapi.com/analyze"
    querystring = {'platform': platform, 'value': value, }
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "dait-reverse-website-lookup-v1.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

