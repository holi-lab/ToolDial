import requests
import json
from datetime import date, datetime, timedelta
import os

from typing import Optional, Dict, Union, List


def single_product_store_prices(barcode: str, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Get all store prices of a product by barcode"
    barcode: This is the barcode found on the product. EAN, UPC, GTIN are accepted.
        
    """
    url = f"https://uk-supermarkets-product-pricing.p.rapidapi.com/product_prices_stores"
    querystring = {'barcode': barcode, }
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "uk-supermarkets-product-pricing.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

