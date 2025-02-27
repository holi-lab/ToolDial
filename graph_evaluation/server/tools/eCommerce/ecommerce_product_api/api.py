import requests
import json
from datetime import date, datetime, timedelta
import os

from typing import Optional, Dict, Union, List


def product_data(product: str, page: int=1, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "The /data endpoint is a part of the Ecommerce Product API that allows you to retrieve information about a specific product in your ecommerce store. By making a GET request to this endpoint and specifying the product name, you can access a wide range of information about the product, including its name, description, images, and more. The response to the request will be in JSON format and will include all relevant product data. This endpoint is an essential tool for retrieving information about products in your store, and is designed to be fast and efficient, ensuring that you receive the information you need quickly and easily."
    
    """
    url = f"https://ecommerce-product-api1.p.rapidapi.com/data"
    querystring = {'product': product, }
    if page:
        querystring['page'] = page
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "ecommerce-product-api1.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

