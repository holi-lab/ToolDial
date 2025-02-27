import requests
import json
from datetime import date, datetime, timedelta
import os

from typing import Optional, Dict, Union, List


def all_latest_rates(toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "All Latest Rates endpoint will return exchange rates based USD. Data is updated every 60 minutes. If you need more updated data (30 / 10 min), contact us."
    
    """
    url = f"https://currency26.p.rapidapi.com/rates"
    querystring = {}
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "currency26.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def latest_rate(is_from: str, to: str, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Latest endpoint will return exchange rate. Data is updated every 60 minutes. If you need more updated data (30 / 10 min), contact us."
    from: Currency from which you are converting.
        to: Currency to which you are converting.
        
    """
    url = f"https://currency26.p.rapidapi.com/rate/{is_from}/{to}"
    querystring = {}
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "currency26.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def convert(value: int, is_from: str, to: str, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Convert endpoint will return converted amount from one currency to another. Data is updated every 60 minutes. If you need more updated data (30 / 10 min), contact us."
    value: The amount to be converted.
        from: Currency from which you are converting.
        to: Currency to which you are converting.
        
    """
    url = f"https://currency26.p.rapidapi.com/convert/{is_from}/{to}/{value}"
    querystring = {}
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "currency26.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def list_of_currencies(toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Endpoint will return list of currencies."
    
    """
    url = f"https://currency26.p.rapidapi.com/list"
    querystring = {}
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "currency26.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

