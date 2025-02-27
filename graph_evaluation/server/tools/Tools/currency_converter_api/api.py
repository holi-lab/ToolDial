import requests
import json
from datetime import date, datetime, timedelta
import os

from typing import Optional, Dict, Union, List


def currency_symbol(code: str, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "The '/currency-symbol' endpoint is feature of the API that allows you to get the symbol of the currency by passing the currency code."
    
    """
    url = f"https://currency-converter-api.p.rapidapi.com/currency-symbol"
    querystring = {'code': code, }
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "currency-converter-api.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def currency_name(code: str, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "The '/currency-name' endpoint is feature of the API that allows you to get the name of the currency by passing the currency code."
    
    """
    url = f"https://currency-converter-api.p.rapidapi.com/currency-name"
    querystring = {'code': code, }
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "currency-converter-api.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def list(toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "The '/list' returns you the available currency. By sending a GET request to this endpoint, you will receive the list of currency as a JSON object."
    
    """
    url = f"https://currency-converter-api.p.rapidapi.com/list"
    querystring = {}
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "currency-converter-api.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def convert(is_from: str, amount: str, to: str, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "The '/convert' endpoint is a powerful feature of the API that allows you to convert an amount of one currency to another in real-time. It takes three query parameters, 'from', 'to' and 'amount', which specify the currencies that you want to convert, and the amount of the 'from' currency. By sending a GET request to this endpoint, you will receive the converted amount in the 'to' currency as a JSON object."
    
    """
    url = f"https://currency-converter-api.p.rapidapi.com/convert"
    querystring = {'from': is_from, 'amount': amount, 'to': to, }
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "currency-converter-api.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def rate(to: str, is_from: str, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "The '/rate' endpoint is a powerful feature of the API that allows you to retrieve the current exchange rate between two currencies in real-time. It takes two query parameters, 'from' and 'to', which specify the currencies that you want to convert. By sending a GET request to this endpoint, you will receive the current exchange rate between the specified currencies as a JSON object."
    
    """
    url = f"https://currency-converter-api.p.rapidapi.com/rate"
    querystring = {'to': to, 'from': is_from, }
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "currency-converter-api.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

