import requests
import json
from datetime import date, datetime, timedelta
import os

from typing import Optional, Dict, Union, List


def historical_exchange_rates(date: str, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Access over two decades of historical exchange rate data with the /historical endpoint. Exchange rates are provided as daily end of day exchange rates dating back all the way to 1999. Choose the day with the 'date' parameter."
    
    """
    url = f"https://exchangeratespro.p.rapidapi.com/historical"
    querystring = {'date': date, }
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "exchangeratespro.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def latest_exchange_rates(resolution: str=None, currencies: str=None, base: str='USD', toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Get up-to-date exchange rate data with the /latest endpoint. Choose from daily, hourly, or minute updates, based on your plan. Stay on top of the markets with real-time data, delivered with one simple request."
    resolution: Select the resolution of the latest exchange rates. 1m will update with new rates every minute, 1h ever hour and 1d every day.
        currencies: Provide a comma separated list of the currencies you want exchange rates for. As default all currencies are returned. ( example: USD,EUR,AUD,GBP )
        base: Change the base currency of the returned rates. USD by default.
        
    """
    url = f"https://exchangeratespro.p.rapidapi.com/latest"
    querystring = {}
    if resolution:
        querystring['resolution'] = resolution
    if currencies:
        querystring['currencies'] = currencies
    if base:
        querystring['base'] = base
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "exchangeratespro.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def list_available_currencies(toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Returns a list of all currencies that are available through the API."
    
    """
    url = f"https://exchangeratespro.p.rapidapi.com/currencies"
    querystring = {}
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "exchangeratespro.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

