import requests
import json
from datetime import date, datetime, timedelta
import os

from typing import Optional, Dict, Union, List


def product_wise_impact(product: str, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "This endpoint provides information about the climate impact of a product based on its full name. This endpoint returns data related to the carbon emissions associated with the product, and breakdown of the impact across various stages of the product's lifecycle."
    
    """
    url = f"https://food-sku-co2-impact-api.p.rapidapi.com/impact/product"
    querystring = {'product': product, }
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "food-sku-co2-impact-api.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def barcode_wise_impact(barcode: str, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "This endpoint provides information about the climate impact of a product based on its barcode. This endpoint returns data related to the carbon emissions associated with the product, and breakdown of the impact across various stages of the product's lifecycle."
    
    """
    url = f"https://food-sku-co2-impact-api.p.rapidapi.com/impact/barcode"
    querystring = {'barcode': barcode, }
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "food-sku-co2-impact-api.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

