import requests
import json
from datetime import date, datetime, timedelta
import os

from typing import Optional, Dict, Union, List


def convert(to: str, amount: str, is_from: str, output: str='JSON', toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Convert a custom amount value from one currency to another"
    to: Currency code of the currency you want to convert to. Eg, USD if you want to convert GBP to USD
        amount: The value you want to convert. Eg 10.99 or 15
        from: Currency code of the currency you want to convert from. Eg, GBP if you want to convert GBP to USD
        output: The output of the response. Either JSON or XML
        
    """
    url = f"https://currencyapi-net.p.rapidapi.com/convert"
    querystring = {'to': to, 'amount': amount, 'from': is_from, }
    if output:
        querystring['output'] = output
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "currencyapi-net.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def currencies(output: str='JSON', toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Gets list of currencies we support"
    output: The output of the response. Either JSON or XML
        
    """
    url = f"https://currencyapi-net.p.rapidapi.com/currencies"
    querystring = {}
    if output:
        querystring['output'] = output
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "currencyapi-net.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def timeframe(end_date: str, base: str, start_date: str, output: str='JSON', toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Display a currency conversion's historical rates within a given timeframe"
    end_date: Display the start date you wish to get the historical data from. Eg 2010-12-29
        base: The currency code you want to get the historical data of. Eg, USD
        start_date: Display the start date you wish to get the historical data from. Eg 2010-12-25
        output: The output of the response. Either JSON or XML
        
    """
    url = f"https://currencyapi-net.p.rapidapi.com/timeframe"
    querystring = {'end_date': end_date, 'base': base, 'start_date': start_date, }
    if output:
        querystring['output'] = output
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "currencyapi-net.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def rates(output: str='JSON', base: str='USD', toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Get live currency conversion rates of a given currency"
    output: The output of the response. Either JSON or XML
        base: Change the base currency. Eg GBP or BTC
        
    """
    url = f"https://currencyapi-net.p.rapidapi.com/rates"
    querystring = {}
    if output:
        querystring['output'] = output
    if base:
        querystring['base'] = base
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "currencyapi-net.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def history(date: str, base: str, output: str='JSON', toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Display the currency conversions historical rates for a particular day"
    date: Display the single date you wish to get the historical data from. Eg 2010-12-25
        base: The currency code you wish to get the historical rates of. Eg, USD
        output: The output of the response. Either JSON or XML
        
    """
    url = f"https://currencyapi-net.p.rapidapi.com/history"
    querystring = {'date': date, 'base': base, }
    if output:
        querystring['output'] = output
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "currencyapi-net.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

