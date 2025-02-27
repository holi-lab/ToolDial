import requests
import json
from datetime import date, datetime, timedelta
import os

from typing import Optional, Dict, Union, List


def smart_phones(name_brand: str=None, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "<br /><br />
		<b>Authentication:</b> not required"
    
    """
    url = f"https://amazon_api.p.rapidapi.com/Smart_Phones"
    querystring = {}
    if name_brand:
        querystring['Name_Brand'] = name_brand
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "amazon_api.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def laptops(name_brand: str=None, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "<br /><br />
		<b>Authentication:</b> not required"
    
    """
    url = f"https://amazon_api.p.rapidapi.com/Laptops"
    querystring = {}
    if name_brand:
        querystring['Name_Brand'] = name_brand
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "amazon_api.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def bcaa(product_name_brand: str=None, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "<br /><br />
		<b>Authentication:</b> not required"
    
    """
    url = f"https://amazon_api.p.rapidapi.com/BCAA"
    querystring = {}
    if product_name_brand:
        querystring['product_name_brand'] = product_name_brand
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "amazon_api.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def headphones(name_brand: str=None, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "<br /><br />
		<b>Authentication:</b> not required"
    
    """
    url = f"https://amazon_api.p.rapidapi.com/Headphones"
    querystring = {}
    if name_brand:
        querystring['Name_Brand'] = name_brand
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "amazon_api.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

