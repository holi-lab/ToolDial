import requests
import json
from datetime import date, datetime, timedelta
import os

from typing import Optional, Dict, Union, List


def get_a_specific_amazon_product(marketplace: str, asin: str, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "This endpoint returns a well formatted amazon product as json. This document contains all product properties, price, buybox, product details and so on."
    marketplace: The marketplace where the product should be obtained. Could be DE, US, UK, JP, ES, IT and so on.
        asin: The products ASIN (amazon product identifier)
        
    """
    url = f"https://sellytics.p.rapidapi.com/amazon/products/{marketplace}/{asin}/v1"
    querystring = {}
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "sellytics.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

