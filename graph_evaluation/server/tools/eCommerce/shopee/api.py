import requests
import json
from datetime import date, datetime, timedelta
import os

from typing import Optional, Dict, Union, List


def seller_details(url: str, key: str, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "The API will fetch sellers details like followers count, ratings, product count, product sold etc. from Shopee website. Working in Malaysia, Singapore, Vietnam, Philippines, and Indonesia domains and languages. [Proxies Intact]"
    
    """
    url = f"https://shopee14.p.rapidapi.com/seller-details/"
    querystring = {'url': url, 'key': key, }
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "shopee14.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

