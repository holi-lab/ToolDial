import requests
import json
from datetime import date, datetime, timedelta
import os

from typing import Optional, Dict, Union, List


def vat_validation(vat_number: str, api_key: str, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "The /validate/ endpoint takes your unique API key and a VAT number in the request, checks if that number is valid, and if it is valid, returns addition details about the company associated with that VAT number."
    
    """
    url = f"https://vat-validation-amp-rates.p.rapidapi.com/"
    querystring = {'vat_number': vat_number, 'api_key': api_key, }
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "vat-validation-amp-rates.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def vat_price_calculation(country_code: str, api_key: str, amount: str, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "The /calculate/ endpoint makes it easy to calculate a VAT compliant price given a country and price, as well as such optional values as the type of goods."
    
    """
    url = f"https://vat-validation-amp-rates.p.rapidapi.com/"
    querystring = {'country_code': country_code, 'api_key': api_key, 'amount': amount, }
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "vat-validation-amp-rates.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def vat_rates(api_key: str, country_code: str, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "The /rates/ endpoint makes it easy to get the latest VAT rates, including the reduced rates for certain categories, for a specific country."
    
    """
    url = f"https://vat-validation-amp-rates.p.rapidapi.com/"
    querystring = {'api_key': api_key, 'country_code': country_code, }
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "vat-validation-amp-rates.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

