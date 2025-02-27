import requests
import json
from datetime import date, datetime, timedelta
import os

from typing import Optional, Dict, Union, List


def latest_rates(base: str='eur', toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "This endpoint will return the exchange rate data updated every 60 minutes."
    base: in case of no base the default is EUR
        
    """
    url = f"https://exchanger-currency-rates-provider.p.rapidapi.com/latest"
    querystring = {}
    if base:
        querystring['base'] = base
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "exchanger-currency-rates-provider.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def supported_currencies(toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Return the list of all supported currencies including 
		
		- Country Name 
		- Country Code
		- Currency Code"
    
    """
    url = f"https://exchanger-currency-rates-provider.p.rapidapi.com/currencies"
    querystring = {}
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "exchanger-currency-rates-provider.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def convert(to: str, is_from: str, amount: str, date: str='2020-01-01', toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "This API with a separate currency conversion endpoint, which can be used to convert any amount from one currency to another. In order to convert currencies, please use the API's convert endpoint, append the from and to parameters and set them to your preferred base and target currency codes."
    
    """
    url = f"https://exchanger-currency-rates-provider.p.rapidapi.com/convert"
    querystring = {'to': to, 'from': is_from, 'amount': amount, }
    if date:
        querystring['date'] = date
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "exchanger-currency-rates-provider.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def historical_rates(date: str='2020-01-01', base: str='eur', toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Request the prices of any date between the year 2000 till today."
    
    """
    url = f"https://exchanger-currency-rates-provider.p.rapidapi.com/historical"
    querystring = {}
    if date:
        querystring['date'] = date
    if base:
        querystring['base'] = base
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "exchanger-currency-rates-provider.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

