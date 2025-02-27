import requests
import json
from datetime import date, datetime, timedelta
import os

from typing import Optional, Dict, Union, List


def price(product: str, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Get the current price from a URL. Keep in mind this can take a while to get depending on how fast the website responds and other things."
    product: The URL of the product you want to get the price for.
        
    """
    url = f"https://pricechecker.p.rapidapi.com/price"
    querystring = {'product': product, }
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "pricechecker.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

