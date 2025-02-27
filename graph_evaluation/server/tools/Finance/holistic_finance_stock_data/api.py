import requests
import json
from datetime import date, datetime, timedelta
import os

from typing import Optional, Dict, Union, List


def forex_symbol(toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Returns a list of all available forex symbols"
    
    """
    url = f"https://holistic-finance-stock-data.p.rapidapi.com/api/v1/symbol/forex"
    querystring = {}
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "holistic-finance-stock-data.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def crypto_symbol(includenames: bool=None, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Returns a list of all available crypto symbols"
    
    """
    url = f"https://holistic-finance-stock-data.p.rapidapi.com/api/v1/symbol/crypto"
    querystring = {}
    if includenames:
        querystring['includeNames'] = includenames
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "holistic-finance-stock-data.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def etf_symbols(includenames: bool=None, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Returns a list of all available etf symbols"
    
    """
    url = f"https://holistic-finance-stock-data.p.rapidapi.com/api/v1/symbol/etf"
    querystring = {}
    if includenames:
        querystring['includeNames'] = includenames
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "holistic-finance-stock-data.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def stock_symbols(includenames: bool=None, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Returns a list of all available stock symbols"
    
    """
    url = f"https://holistic-finance-stock-data.p.rapidapi.com/api/v1/symbol/stock"
    querystring = {}
    if includenames:
        querystring['includeNames'] = includenames
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "holistic-finance-stock-data.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def historical_crypto_price(source: str, target: str, interval: str, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Provides historical crypto currency data."
    
    """
    url = f"https://holistic-finance-stock-data.p.rapidapi.com/api/v1/historical/crypto"
    querystring = {'source': source, 'target': target, 'interval': interval, }
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "holistic-finance-stock-data.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def historical_forex_price(target: str, source: str, interval: str, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Provides historical forex data."
    
    """
    url = f"https://holistic-finance-stock-data.p.rapidapi.com/api/v1/historical/forex"
    querystring = {'target': target, 'source': source, 'interval': interval, }
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "holistic-finance-stock-data.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def historical_stock_price(symbol: str, interval: str, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Provides historical stock data."
    
    """
    url = f"https://holistic-finance-stock-data.p.rapidapi.com/api/v1/historical/stock"
    querystring = {'symbol': symbol, 'interval': interval, }
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "holistic-finance-stock-data.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def crypto_quotes(source: str, target: str, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Provides real-time crypto data."
    
    """
    url = f"https://holistic-finance-stock-data.p.rapidapi.com/api/v1/realtime/crypto"
    querystring = {'source': source, 'target': target, }
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "holistic-finance-stock-data.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def forex_quotes(source: str, target: str, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Provides real-time forex data."
    
    """
    url = f"https://holistic-finance-stock-data.p.rapidapi.com/api/v1/realtime/forex"
    querystring = {'source': source, 'target': target, }
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "holistic-finance-stock-data.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def stock_quotes(symbols: str, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Provides real-time stock data of a single or multiple stock companies."
    
    """
    url = f"https://holistic-finance-stock-data.p.rapidapi.com/api/v1/realtime/stock"
    querystring = {'symbols': symbols, }
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "holistic-finance-stock-data.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def earnings(symbol: str, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Provides earnings of a single stock company."
    
    """
    url = f"https://holistic-finance-stock-data.p.rapidapi.com/api/v1/eps"
    querystring = {'symbol': symbol, }
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "holistic-finance-stock-data.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def estimates(symbol: str, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Provides estimates of a single stock company."
    
    """
    url = f"https://holistic-finance-stock-data.p.rapidapi.com/api/v1/estimates"
    querystring = {'symbol': symbol, }
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "holistic-finance-stock-data.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def grades(symbol: str, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Provides grades of a single stock company."
    
    """
    url = f"https://holistic-finance-stock-data.p.rapidapi.com/api/v1/grade"
    querystring = {'symbol': symbol, }
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "holistic-finance-stock-data.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def scores(symbol: str, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Provides scores of a single stock company."
    
    """
    url = f"https://holistic-finance-stock-data.p.rapidapi.com/api/v1/scores"
    querystring = {'symbol': symbol, }
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "holistic-finance-stock-data.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def key_metrics(symbol: str, period: str, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Provides either annual or quarterly key metrics of a single stock company."
    
    """
    url = f"https://holistic-finance-stock-data.p.rapidapi.com/api/v1/keymetrics"
    querystring = {'symbol': symbol, 'period': period, }
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "holistic-finance-stock-data.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def ratios(symbol: str, period: str, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Provides either annual or quarterly ratios of a single stock company."
    
    """
    url = f"https://holistic-finance-stock-data.p.rapidapi.com/api/v1/ratios"
    querystring = {'symbol': symbol, 'period': period, }
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "holistic-finance-stock-data.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def stock_ownership(symbol: str, includecurrentquarter: bool, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Provides current stock ownership information."
    
    """
    url = f"https://holistic-finance-stock-data.p.rapidapi.com/api/v1/ownership"
    querystring = {'symbol': symbol, 'includeCurrentQuarter': includecurrentquarter, }
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "holistic-finance-stock-data.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def holder(symbol: str, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Provides information about the current holder structure."
    
    """
    url = f"https://holistic-finance-stock-data.p.rapidapi.com/api/v1/holder"
    querystring = {'symbol': symbol, }
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "holistic-finance-stock-data.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def executives(symbol: str, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Provides information about the current composition of executives"
    
    """
    url = f"https://holistic-finance-stock-data.p.rapidapi.com/api/v1/executives"
    querystring = {'symbol': symbol, }
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "holistic-finance-stock-data.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def shares_float(symbol: str, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Provides the amount of current shares float."
    
    """
    url = f"https://holistic-finance-stock-data.p.rapidapi.com/api/v1/sharesfloat"
    querystring = {'symbol': symbol, }
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "holistic-finance-stock-data.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def profile(symbol: str, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Provides company data e.g. sector, industry, location, executives..."
    
    """
    url = f"https://holistic-finance-stock-data.p.rapidapi.com/api/v1/profile"
    querystring = {'symbol': symbol, }
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "holistic-finance-stock-data.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def income(symbol: str, period: str, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Provides either annual or quarterly income statements of an single stock company."
    
    """
    url = f"https://holistic-finance-stock-data.p.rapidapi.com/api/v1/income"
    querystring = {'symbol': symbol, 'period': period, }
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "holistic-finance-stock-data.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def cashflow(symbol: str, period: str, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Provides either annual or quarterly cash flow statements of an single stock company."
    
    """
    url = f"https://holistic-finance-stock-data.p.rapidapi.com/api/v1/cashflow"
    querystring = {'symbol': symbol, 'period': period, }
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "holistic-finance-stock-data.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def balance(symbol: str, period: str, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Provides either annual or quarterly balance sheet statements of an single stock company."
    
    """
    url = f"https://holistic-finance-stock-data.p.rapidapi.com/api/v1/balancesheets"
    querystring = {'symbol': symbol, 'period': period, }
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "holistic-finance-stock-data.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

