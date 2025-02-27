import requests
import json
from datetime import date, datetime, timedelta
import os

from typing import Optional, Dict, Union, List


def get_4_date_endpoint(base: str, symbols: str, date: str='2023-02-01', toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Request Parameters:
		'date': 
		'symbols': 
		'base': 
		Response: This endpoint returns the exchange rates for the specified currencies on the specified date in JSON format."
    
    """
    url = f"https://currenciesexchangerateapi.p.rapidapi.com/date"
    querystring = {'base': base, 'symbols': symbols, }
    if date:
        querystring['date'] = date
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "currenciesexchangerateapi.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def get_6_timeseries_endpoint(end_date: str, start_date: str, base: str='usd', symbols: str='eur', toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Request Parameters:
		'start_date': 
		'end_date': 
		'base': 
		'symbols': 
		Response: This endpoint returns the exchange rates for the specified currencies over the specified time period in JSON format."
    
    """
    url = f"https://currenciesexchangerateapi.p.rapidapi.com/timeseries"
    querystring = {'end_date': end_date, 'start_date': start_date, }
    if base:
        querystring['base'] = base
    if symbols:
        querystring['symbols'] = symbols
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "currenciesexchangerateapi.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def get_5_latest_endpoint(symbols: str, base: str, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Endpoint URL:/latest?symbols=EUR&base=usd
		HTTP Method: GET
		Request Parameters:
		'symbols': 
		'base': 
		Response: This endpoint returns the latest exchange rates for the specified currencies in JSON format."
    
    """
    url = f"https://currenciesexchangerateapi.p.rapidapi.com/latest"
    querystring = {'symbols': symbols, 'base': base, }
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "currenciesexchangerateapi.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def get_3_fluctuation_endpoint(end_date: str, base: str, start_date: str, symbols: str='usd', toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Request Parameters:
		'start_date': 
		'end_date': 
		'base': 
		'symbols': 
		Response: This endpoint returns the percentage change in the exchange rates of the specified currencies over the specified time period in JSON format."
    
    """
    url = f"https://currenciesexchangerateapi.p.rapidapi.com/fluctuation"
    querystring = {'end_date': end_date, 'base': base, 'start_date': start_date, }
    if symbols:
        querystring['symbols'] = symbols
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "currenciesexchangerateapi.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def get_2_convert_endpoint(amount: int, is_from: str, to: str, date: str=None, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Endpoint URL: /convert?to=USD&from=EUR&amount=100&date=2023-02-01
		HTTP Method: GET
		Request Parameters:
		'to': The target currency symbol. (Required)
		'from': The source currency symbol. (Required)
		'amount': The amount to be converted. (Required)
		'date': 
		Response: This endpoint returns the converted amount and the exchange rate used for the conversion in JSON format."
    
    """
    url = f"https://currenciesexchangerateapi.p.rapidapi.com/convert"
    querystring = {'amount': amount, 'from': is_from, 'to': to, }
    if date:
        querystring['date'] = date
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "currenciesexchangerateapi.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def get_1_availablesymbols_endpoint(toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Endpoint URL: /availablesymbols
		HTTP Method: GET
		Response: This endpoint returns a list of all the available currency symbols that can be used for conversion in JSON format.
		supports 170 symbols for currencies"
    
    """
    url = f"https://currenciesexchangerateapi.p.rapidapi.com/availablesymbols"
    querystring = {}
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "currenciesexchangerateapi.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

