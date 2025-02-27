import requests
import json
from datetime import date, datetime, timedelta
import os

from typing import Optional, Dict, Union, List


def open_high_low_close_ohlc_price(symbols: str, date: str, base: str, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Allows you to query the API to get the open, high, low, and close price.
		This endpoint has a limitation of one symbol per request."
    symbols: Enter the three-letter currency code or commodity code of your preferred base currency
        base: Enter the three-letter currency code or commodity code of your preferred base currency
        
    """
    url = f"https://commodity-rates-api.p.rapidapi.com/open-high-low-close/{date}"
    querystring = {'symbols': symbols, 'base': base, }
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "commodity-rates-api.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def time_series(base: str, symbols: str, end_date: str, start_date: str, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "With this endpoint, you will be provided with daily historical rates between two dates of your choice. 
		
		Note: This endpoint has a limitation of 365 days per range, and only can provide one symbol information per request.
		
		All the commodities rates you get need using ANY currency as a base currency (base parameter) need to be divided by 1
		We return the values based on the base currency. For example, for 1 USD the return is a number like 0.00055307742 for Gold (XAU).
		To get the gold rate per troy ounce in USD: 1/0.00055307742 = 1808.06 USD"
    base: Enter the three-letter currency code or commodity code of your preferred base currency
        symbols: Enter the three-letter currency code or commodity code of your preferred base currency
        end_date: The end date of your preferred timeframe.
        start_date: The start date of your preferred timeframe.
        
    """
    url = f"https://commodity-rates-api.p.rapidapi.com/timeseries"
    querystring = {'base': base, 'symbols': symbols, 'end_date': end_date, 'start_date': start_date, }
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "commodity-rates-api.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def historical_rates(base: str, symbols: str, date: str, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Historical rates are available for most symbols all the way back to the year 2021. You can query the API for historical rates by appending a date (format YYYY-MM-DD) to the base URL.
		
		Note: All the commodities rates you get need using ANY currency as a base currency (base parameter) need to be divided by 1
		We return the values based on the base currency. For example, for 1 USD the return is a number like 0.00055307742 for Gold (XAU).
		To get the gold rate per troy ounce in USD: 1/0.00055307742 = 1808.06 USD"
    
    """
    url = f"https://commodity-rates-api.p.rapidapi.com/{date}"
    querystring = {'base': base, 'symbols': symbols, }
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "commodity-rates-api.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def latest_rates(symbols: str, base: str, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Check the real-time exchange rate data updated every 60 seconds with this endpoint. 
		 
		
		Note: All the commodities rates you get need using ANY currency as a base currency (base parameter) need to be divided by 1
		We return the values based on the base currency. For example, for 1 USD the return is a number like 0.00055307742 for Gold (XAU).
		To get the gold rate per troy ounce in USD: 1/0.00055307742 = 1808.06 USD"
    symbols: Enter a list of comma-separated currency codes or commodity codes to limit output codes. Check Symbols endpoint
        base: Enter the three-letter currency code or commodity code of your preferred base currency. Check Symbols endpoint
        
    """
    url = f"https://commodity-rates-api.p.rapidapi.com/latest"
    querystring = {'symbols': symbols, 'base': base, }
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "commodity-rates-api.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def supported_symbols(toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Check all the available symbols for your queries. This endpoint retrieves all the supported commodities and currencies as well."
    
    """
    url = f"https://commodity-rates-api.p.rapidapi.com/symbols"
    querystring = {}
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "commodity-rates-api.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

