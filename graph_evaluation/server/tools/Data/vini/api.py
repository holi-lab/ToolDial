import requests
import json
from datetime import date, datetime, timedelta
import os

from typing import Optional, Dict, Union, List


def balance(key: str, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Check your balance"
    key: Your API key
        
    """
    url = f"https://vini.p.rapidapi.com/balance"
    querystring = {'key': key, }
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "vini.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def register(email: str, name: str='John Smith', toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Register an API key"
    email: Your email address
        name: Your full name, i.e. John Smith
        
    """
    url = f"https://vini.p.rapidapi.com/register"
    querystring = {'email': email, }
    if name:
        querystring['name'] = name
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "vini.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def check(type: str, key: str, vin: str, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Check report availability by VIN code"
    type: Report type: carfax, autocheck, copart
        key: Your API key
        vin: VIN code
        
    """
    url = f"https://vini.p.rapidapi.com/check"
    querystring = {'type': type, 'key': key, 'vin': vin, }
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "vini.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def report(key: str, type: str, vin: str, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Generate and download report"
    key: Your API key
        type: Report provider: carfax, autocheck, copart
        vin: VIN code
        
    """
    url = f"https://vini.p.rapidapi.com/report"
    querystring = {'key': key, 'type': type, 'vin': vin, }
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "vini.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def replenish(amount: int, key: str, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Manually add funds to balance"
    amount: Amount you want to add to balance. After request you'll be redirected to our bank's payment page.
        key: Your API key
        
    """
    url = f"https://vini.p.rapidapi.com/replenish"
    querystring = {'amount': amount, 'key': key, }
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "vini.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

