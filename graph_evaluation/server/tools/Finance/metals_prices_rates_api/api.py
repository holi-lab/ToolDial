import requests
import json
from datetime import date, datetime, timedelta
import os

from typing import Optional, Dict, Union, List


def open_high_low_close_ohlc_price(symbols: str, base: str, date: str, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Allows you to query the API to get the open, high, low, and close price.
		This endpoint has a limitation of one symbol per request."
    
    """
    url = f"https://metals-prices-rates-api.p.rapidapi.com/open-high-low-close/{date}"
    querystring = {'symbols': symbols, 'base': base, }
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "metals-prices-rates-api.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def carat(toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Using the API karat endpoint, you will be able to retrieve latest information about gold rates per karat."
    
    """
    url = f"https://metals-prices-rates-api.p.rapidapi.com/carat"
    querystring = {}
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "metals-prices-rates-api.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def time_series(base: str, symbols: str, end_date: str, start_date: str, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Timeseries endpoint lets you query the API for daily historical rates between two dates of your choice.
		This endpoint has a limitation of 365 days and only one symbol per request."
    base: Enter the three-letter currency code or metal code of your preferred base currency.
        symbols: Enter the three-letter currency code or metal code of your preferred base currency.
        end_date: The end date of your preferred timeframe. YYYY-MM-DD
        start_date: The start date of your preferred timeframe. YYYY-MM-DD
        
    """
    url = f"https://metals-prices-rates-api.p.rapidapi.com/timeseries"
    querystring = {'base': base, 'symbols': symbols, 'end_date': end_date, 'start_date': start_date, }
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "metals-prices-rates-api.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def supported_symbols(toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "The API comes with a constantly updated endpoint returning all available symbols."
    
    """
    url = f"https://metals-prices-rates-api.p.rapidapi.com/symbols"
    querystring = {}
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "metals-prices-rates-api.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def historical_rates(date: str, symbols: str, base: str, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Historical rates are available for most symbols all the way back to the year of 2019. You can query the API for historical rates by appending a date (format YYYY-MM-DD) to the base URL."
    symbols: Enter a list of comma-separated currency codes or metal codes to limit output codes.
        base: Enter the three-letter currency code or metal code of your preferred base currency.
        
    """
    url = f"https://metals-prices-rates-api.p.rapidapi.com/{date}"
    querystring = {'symbols': symbols, 'base': base, }
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "metals-prices-rates-api.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def latest_rates(symbols: str, base: str, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "The latest API endpoint will return real-time exchange rate data updated every 60 seconds.
		
		Note: All the responses retrieved in USD needs to be converted. 
		
		Ex: 1/value = USD PRICE."
    symbols: Enter a list of comma-separated currency codes or metal codes to limit output codes.
        base: Enter the three-letter currency code or metal code of your preferred base currency.
        
    """
    url = f"https://metals-prices-rates-api.p.rapidapi.com/latest"
    querystring = {'symbols': symbols, 'base': base, }
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "metals-prices-rates-api.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

