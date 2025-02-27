import requests
import json
from datetime import date, datetime, timedelta
import os

from typing import Optional, Dict, Union, List


def get_all_cryptocurrencies_supported_short_name_with_the_full_name(toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Fetch a list of supported crypto currencies."
    
    """
    url = f"https://currency-conveter-api-unlimited-reqests.p.rapidapi.com/crypto/currencies"
    querystring = {}
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "currency-conveter-api-unlimited-reqests.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def fetch_all_available_cryptocurrencies_rates_from_one_to_all(is_from: str, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Fetch all available Crypto Currencies rates from Physical or Cryptocurrencies."
    from: Base currency symbol (physical or digital).
        
    """
    url = f"https://currency-conveter-api-unlimited-reqests.p.rapidapi.com/crypto/fetchAll"
    querystring = {'from': is_from, }
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "currency-conveter-api-unlimited-reqests.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def get_all_currencies_supported_short_name_with_the_full_name(toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Fetch a list of supported physical currencies."
    
    """
    url = f"https://currency-conveter-api-unlimited-reqests.p.rapidapi.com/currencies"
    querystring = {}
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "currency-conveter-api-unlimited-reqests.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def fetch_all_available_physical_currency_rates_from_one_to_all(is_from: str, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Fetch all available Physical Currencies rates from Physical or Cryptocurrencies."
    from: Base currency symbol (physical or digital).
        
    """
    url = f"https://currency-conveter-api-unlimited-reqests.p.rapidapi.com/fetchAll"
    querystring = {'from': is_from, }
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "currency-conveter-api-unlimited-reqests.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def fetch_a_single_currency_rate(to: str, is_from: str, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Fetch a single currency exchange rate, from and to any supported physical or digital currency.
		
		For example, you can use it to fetch from **GBP** to **USD**."
    
    """
    url = f"https://currency-conveter-api-unlimited-reqests.p.rapidapi.com/fetchOne"
    querystring = {'to': to, 'from': is_from, }
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "currency-conveter-api-unlimited-reqests.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def fetch_multi_currency_rate(to: str, is_from: str, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Fetch multi currency exchange rate, from and to any supported physical or digital currency.
		
		For example, you can use it to fetch from GBP to **USD**, **EUR** and **AUD** with one request..."
    to: Target multi currencies symbol (physical or digital or mix).
Example: GBP,EUR,AUD
        from: Base currency symbol (physical or digital).
        
    """
    url = f"https://currency-conveter-api-unlimited-reqests.p.rapidapi.com/fetchMulti"
    querystring = {'to': to, 'from': is_from, }
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "currency-conveter-api-unlimited-reqests.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def currency_converter(to: str, amount: int, is_from: str, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Convert an amount of one currency into another currency (supports physical and digital currencies).
		
		For example, you can use it to convert **10 GBP** to **AUD**"
    to: Target multi currencies symbol (physical or digital).
        amount: Amount of source currency to convert.
        from: Base currency symbol (physical or digital).
        
    """
    url = f"https://currency-conveter-api-unlimited-reqests.p.rapidapi.com/convert"
    querystring = {'to': to, 'amount': amount, 'from': is_from, }
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "currency-conveter-api-unlimited-reqests.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

