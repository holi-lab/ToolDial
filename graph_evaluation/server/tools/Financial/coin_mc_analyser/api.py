import requests
import json
from datetime import date, datetime, timedelta
import os

from typing import Optional, Dict, Union, List


def get_all_crypto_currencies_with_loss_above_limit_within_the_last_7d(limit: str, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "This endpoint fetches all cryptocurrencies which values have depreciated above a specified limit (number with 2 decimal places) with the last 7 days"
    
    """
    url = f"https://coin-mc-analyser.p.rapidapi.com/currencies/7d/loss/{limit}"
    querystring = {}
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "coin-mc-analyser.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def get_all_crypto_currencies_with_loss_above_limit_within_the_last_24hours(limit: str, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "This endpoint fetches all cryptocurrencies which values have depreciated above a specified limit (number with 2 decimal places) with the last 24hrs"
    
    """
    url = f"https://coin-mc-analyser.p.rapidapi.com/currencies/24h/loss/{limit}"
    querystring = {}
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "coin-mc-analyser.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def get_all_crypto_currencies_with_profit_above_limit_within_the_last_7_days(limit: str, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "This endpoint fetches all cryptocurrencies which values have grown above a specified limit (number with 2 decimal places) with the last 7 days."
    
    """
    url = f"https://coin-mc-analyser.p.rapidapi.com/currencies/7d/profit/{limit}"
    querystring = {}
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "coin-mc-analyser.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def get_all_crypto_currencies_with_profit_above_limit_within_the_last_24hours(limit: str, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "This endpoint fetches all cryptocurrencies which values have grown above a specified limit (number with 2 decimal places) with the last 24hrs."
    
    """
    url = f"https://coin-mc-analyser.p.rapidapi.com/currencies/24h/profit/{limit}"
    querystring = {}
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "coin-mc-analyser.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def get_individual_crypto_currency(coinid: str, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "This endpoint returns more specific but extensive stat data for a particular cryptocurrency."
    
    """
    url = f"https://coin-mc-analyser.p.rapidapi.com/currencies/{coinid}"
    querystring = {}
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "coin-mc-analyser.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def get_all_crypto_currencies(toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "This endpoint will return the top 100 cryptocurrencies and their statistics"
    
    """
    url = f"https://coin-mc-analyser.p.rapidapi.com/currencies"
    querystring = {}
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "coin-mc-analyser.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

