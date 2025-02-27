import requests
import json
from datetime import date, datetime, timedelta
import os

from typing import Optional, Dict, Union, List


def main_endpoint(url: str, path: str, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "It Gives you the product’s information 
		
		The Path for every store:
		Amazon = am
		Walmart = wal
		eBay = eb
		Bestbuy = best"
    
    """
    url = f"https://amazon-and-walmart-and-ebay-and-bestbuy.p.rapidapi.com/index.php"
    querystring = {'url': url, 'path': path, }
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "amazon-and-walmart-and-ebay-and-bestbuy.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

