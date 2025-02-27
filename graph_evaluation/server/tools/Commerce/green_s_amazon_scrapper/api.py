import requests
import json
from datetime import date, datetime, timedelta
import os

from typing import Optional, Dict, Union, List


def amazon_product_details(productid: str, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Amazon Product Details. You have to first go to scraperapi.com to obtain a unique API key before you can proceed. After registering, you will have 5,000  point to request with."
    
    """
    url = f"https://greens-amazon-scrapper.p.rapidapi.com/product/{productid}"
    querystring = {}
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "greens-amazon-scrapper.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

