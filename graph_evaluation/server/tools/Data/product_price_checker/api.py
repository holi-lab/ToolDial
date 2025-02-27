import requests
import json
from datetime import date, datetime, timedelta
import os

from typing import Optional, Dict, Union, List


def get_current_product_price(ean: str, market: str, method: str, apikey: str, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "This method allows you to know the price of a product in different stores in a given country, using the EAN of the product."
    ean: Product EAN identifier
        market: Country in which you want to check prices, the possible values are: uk, fr, it, de and es
        method: API method called, in this case product.getprice
        apikey: API Key of the user, you need to access your ConsumerStore account to consult yours
        
    """
    url = f"https://product-price-checker.p.rapidapi.com/"
    querystring = {'ean': ean, 'market': market, 'method': method, 'apikey': apikey, }
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "product-price-checker.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

