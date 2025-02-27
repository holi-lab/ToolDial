import requests
import json
from datetime import date, datetime, timedelta
import os

from typing import Optional, Dict, Union, List


def get_a_specific_food_product_by_gtin(gtin: str, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Returns information about a specific food product identified by its GTIN, including its title, image URL, net weight, category, ingredients, allergens, nutritional information, and manufacturer details."
    
    """
    url = f"https://nutrinet-api.p.rapidapi.com/products/{gtin}"
    querystring = {}
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "nutrinet-api.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def get_all_food_products(toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Returns a list of all food products in the database, including their GTIN, title, image URL, net weight, category, ingredients, allergens, nutritional information, and manufacturer details."
    
    """
    url = f"https://nutrinet-api.p.rapidapi.com/products"
    querystring = {}
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "nutrinet-api.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

