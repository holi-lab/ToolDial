import requests
import json
from datetime import date, datetime, timedelta
import os

from typing import Optional, Dict, Union, List


def getforecastllist_mega(toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Get Full Crypto Asset List with Expected Daily Forecast Change"
    
    """
    url = f"https://the-apophis.p.rapidapi.com/assets/full_list"
    querystring = {}
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "the-apophis.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def getforecast_ultra(timeframe: str, currency: str, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "This endpoint is part of Ultra subscriptions.
		
		**Available Currencies:**
		All traded coins & tokens on Coinbase.
		
		**Forecast Data:**
		- 10 Periods
		
		**Timeframe:**
		- 1m
		- 5m
		- 15m
		- 1h
		- 6h"
    
    """
    url = f"https://the-apophis.p.rapidapi.com/forecast/ultra"
    querystring = {'timeframe': timeframe, 'currency': currency, }
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "the-apophis.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def getforecast_mega(timeframe: str, currency: str, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "This endpoint is part of Mega subscriptions.
		
		**Available Currencies:**
		All traded coins & tokens on Coinbase.
		
		**Forecast Data:**
		- 30 Periods
		
		**Timeframe:**
		- 1m
		- 5m
		- 15m
		- 1h
		- 6h"
    
    """
    url = f"https://the-apophis.p.rapidapi.com/forecast/mega"
    querystring = {'timeframe': timeframe, 'currency': currency, }
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "the-apophis.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def getforecast_pro(currency: str, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "This endpoint is part of Pro subscriptions.
		
		**Available Currencies:**
		All traded coins & tokens on Coinbase.
		
		**Forecast Data:**
		- 10 Periods
		
		**Timeframe:**
		- 1h"
    
    """
    url = f"https://the-apophis.p.rapidapi.com/forecast/pro"
    querystring = {'currency': currency, }
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "the-apophis.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def getassetslist_pro(toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Retrieve a list of available assets for forecasting"
    
    """
    url = f"https://the-apophis.p.rapidapi.com/assets"
    querystring = {}
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "the-apophis.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def getforecast_free(currency: str, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "This endpoint is part of Basic (Free) subscriptions.
		
		**Available Currencies:**
		- BTC
		- ETH
		
		**Forecast Data:**
		- 3 Periods
		
		**Timeframe:**
		- 1h"
    
    """
    url = f"https://the-apophis.p.rapidapi.com/forecast/free"
    querystring = {'currency': currency, }
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "the-apophis.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

