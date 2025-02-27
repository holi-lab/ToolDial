import requests
import json
from datetime import date, datetime, timedelta
import os

from typing import Optional, Dict, Union, List


def historical(api_key: str, base: str='USDx', quotes: str='BTC', toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "This endpoint will return historical exchange rate data. Use format YYYY-MM-DD for the base url."
    
    """
    url = f"https://metalpriceapi.p.rapidapi.com/v1/2014-10-14"
    querystring = {'api_key': api_key, }
    if base:
        querystring['base'] = base
    if quotes:
        querystring['quotes'] = quotes
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "metalpriceapi.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def change(api_key: str, start_date: str, end_date: str, currencies: str='EUR,XAU,XAG', base: str='USD', toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "This endpoint is used to request the change (percentage) of one of more currencies, relative to a base currency with a specific time-frame (optional)."
    
    """
    url = f"https://metalpriceapi.p.rapidapi.com/v1/change"
    querystring = {'api_key': api_key, 'start_date': start_date, 'end_date': end_date, }
    if currencies:
        querystring['currencies'] = currencies
    if base:
        querystring['base'] = base
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "metalpriceapi.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def live(api_key: str, currencies: str='XAU,XAG,EUR', base: str='USD', toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "This endpoint will return real-time exchange rate data"
    
    """
    url = f"https://metalpriceapi.p.rapidapi.com/v1/latest"
    querystring = {'api_key': api_key, }
    if currencies:
        querystring['currencies'] = currencies
    if base:
        querystring['base'] = base
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "metalpriceapi.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def convert(to: str, api_key: str, amount: int, is_from: str, date: str='2021-03-10', toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "This endpoint is used to convert any amount from one currency to another."
    
    """
    url = f"https://metalpriceapi.p.rapidapi.com/v1/convert"
    querystring = {'to': to, 'api_key': api_key, 'amount': amount, 'from': is_from, }
    if date:
        querystring['date'] = date
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "metalpriceapi.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def timeframe(api_key: str, end_date: str, start_date: str, currencies: str='EUR,XAU,XAG', base: str='USD', toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "This endpoint is used to get historical rates for a specified time period (max range: 365 days)."
    
    """
    url = f"https://metalpriceapi.p.rapidapi.com/v1/timeframe"
    querystring = {'api_key': api_key, 'end_date': end_date, 'start_date': start_date, }
    if currencies:
        querystring['currencies'] = currencies
    if base:
        querystring['base'] = base
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "metalpriceapi.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def symbols(api_key: str, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "This endpoint is used to get the list of the most up-to-date supported symbols."
    
    """
    url = f"https://metalpriceapi.p.rapidapi.com/v1/symbols"
    querystring = {'api_key': api_key, }
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "metalpriceapi.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

