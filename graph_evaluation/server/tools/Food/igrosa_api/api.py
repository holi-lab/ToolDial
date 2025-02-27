import requests
import json
from datetime import date, datetime, timedelta
import os

from typing import Optional, Dict, Union, List


def pick_n_pay_product_information_endpoint(productid: str, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "This endpoint retrieves product information from Pick 'n Pay (supermarket identifier) and productId (product identifier) in the URL parameters. It returns a JSON response containing details such as the product name, price, image URL, and the name of the supermarket."
    
    """
    url = f"https://igrosa-api.p.rapidapi.com/pnp/{productid}"
    querystring = {}
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "igrosa-api.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def balmoral_product_information_endpoint(productid: str, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "This endpoint retrieves product information from Balmoral (supermarket identifier) and productId (product identifier) in the URL parameters. It returns a JSON response containing details such as the product name, price, image URL, and the name of the supermarket."
    
    """
    url = f"https://igrosa-api.p.rapidapi.com/balmoral/{productid}"
    querystring = {}
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "igrosa-api.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def checkers_product_information_endpoint(productid: str, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "This endpoint retrieves product information from Checkers (supermarket identifier) and productId (product identifier) in the URL parameters. It returns a JSON response containing details such as the product name, price, image URL, and the name of the supermarket."
    
    """
    url = f"https://igrosa-api.p.rapidapi.com/checkers/{productid}"
    querystring = {}
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "igrosa-api.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def shoprite_product_information_endpoint(productid: str, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "This endpoint retrieves product information from Shoprite (supermarket identifier) and productId (product identifier) in the URL parameters. It returns a JSON response containing details such as the product name, price, image URL, and the name of the supermarket."
    
    """
    url = f"https://igrosa-api.p.rapidapi.com/shoprite/{productid}"
    querystring = {}
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "igrosa-api.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def root_endpoint(toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "This endpoint provides a welcome message."
    
    """
    url = f"https://igrosa-api.p.rapidapi.com/"
    querystring = {}
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "igrosa-api.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

