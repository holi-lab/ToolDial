import requests
import json
from datetime import date, datetime, timedelta
import os

from typing import Optional, Dict, Union, List


def convert(is_from: str='EUR', to: str='USD', amount: str='1', toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "You can convert the currencies and get a value."
    
    """
    url = f"https://currencies-and-countries.p.rapidapi.com/convert"
    querystring = {}
    if is_from:
        querystring['from'] = is_from
    if to:
        querystring['to'] = to
    if amount:
        querystring['amount'] = amount
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "currencies-and-countries.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def get_country_info(value: str='DE', param: str='ISO', toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "You can get country info by using countryName or Iso Name (DE).
		Example :
		
		```
		                       params: {param: 'iso', value: 'DE'}
		                       params: {param: 'countryName', value: 'Germany'}
		```
		
		These two is gets the same result which is the information of germany. 
		Example response:
		
		```
		{"iso":"DE"
		"currency":"EUR"
		"symbol":"€"
		"countryName":"Germany"
		"dateFormat":"dd.MM.yyyy"}
		```
		
		Example2:
		
		```
		                       params: {param: 'currency', value: 'USD'}
		                       params: {param: 'symbol', value: '$'}
		```
		
		You will get a response like this below. It contains all countries that are using  defined currency. 
		
		Example2 Response:
		
		```
		  {
		    iso: 'US',
		    currency: 'USD',
		    symbol: '$',
		    countryName: 'United States',
		    dateFormat: 'M/d/yyyy'
		  },
		  . . .
		```"
    
    """
    url = f"https://currencies-and-countries.p.rapidapi.com/getCountryInfo"
    querystring = {}
    if value:
        querystring['value'] = value
    if param:
        querystring['param'] = param
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "currencies-and-countries.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def convert_get_symbol(amount: str='1', to: str='USD', is_from: str='EUR', toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "It is the same thing with the convert endpoint. But this one can give you the symbol of the converted value."
    
    """
    url = f"https://currencies-and-countries.p.rapidapi.com/convertWithSymbol"
    querystring = {}
    if amount:
        querystring['amount'] = amount
    if to:
        querystring['to'] = to
    if is_from:
        querystring['from'] = is_from
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "currencies-and-countries.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def crypto_convert(is_from: str='ETH', to: str='BTC', amount: str='100', toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "You can convert crypto currencies."
    
    """
    url = f"https://currencies-and-countries.p.rapidapi.com/cryptoConvert"
    querystring = {}
    if is_from:
        querystring['from'] = is_from
    if to:
        querystring['to'] = to
    if amount:
        querystring['amount'] = amount
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "currencies-and-countries.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

