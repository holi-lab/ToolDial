import requests
import json
from datetime import date, datetime, timedelta
import os

from typing import Optional, Dict, Union, List


def option_contract_specs_by_trading_symbol(symbol: str, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Access specifications of a particular option contract by providing its trading symbol.  Returned data contains underlying asset, option type (call or put), exercise price and expiration date."
    
    """
    url = f"https://greek-financial-derivatives-market-historical-data.p.rapidapi.com/optionsymbol/"
    querystring = {'symbol': symbol, }
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "greek-financial-derivatives-market-historical-data.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def option_contract_underlying_assets(toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Access full list of underlying assets (stocks, indices) for all option contracts."
    
    """
    url = f"https://greek-financial-derivatives-market-historical-data.p.rapidapi.com/optionasset/"
    querystring = {}
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "greek-financial-derivatives-market-historical-data.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def future_contract_underlying_assets(toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Access full list of underlying assets (stocks, indices) for all future contracts."
    
    """
    url = f"https://greek-financial-derivatives-market-historical-data.p.rapidapi.com/futureasset/"
    querystring = {}
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "greek-financial-derivatives-market-historical-data.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def future_contract_specs_by_asset_expiration(asset: str, expmonthdate: str='2022-04-15', toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Access specifications of future contracts by providing their underlying asset and expiration date as optional parameters.  Returned data contains underlying asset and expiration date for each contract. Expiration date (3rd Friday of each month) must be in format YYYY-MM-DD."
    
    """
    url = f"https://greek-financial-derivatives-market-historical-data.p.rapidapi.com/futuresymbolasset/"
    querystring = {'asset': asset, }
    if expmonthdate:
        querystring['expmonthdate'] = expmonthdate
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "greek-financial-derivatives-market-historical-data.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def option_contract_specs_by_asset_optiontype_expiration(asset: str, optiontype: str, expmonthdate: str='2022-04-15', toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Access specifications of option contracts by providing their underlying asset, option type and expiration date as optional parameters.  Returned data contains underlying asset, option type (call or put), exercise price and expiration date for each contract. Use "c" for call and "p" for put options. Expiration date (3rd Friday of each month) must be in format YYYY-MM-DD."
    
    """
    url = f"https://greek-financial-derivatives-market-historical-data.p.rapidapi.com/optionsymbolasset/"
    querystring = {'asset': asset, 'optiontype': optiontype, }
    if expmonthdate:
        querystring['expmonthdate'] = expmonthdate
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "greek-financial-derivatives-market-historical-data.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def future_contract_specs_by_trading_symbol(symbol: str, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Access specifications of a particular future contract by providing its trading symbol.  Returned data contains underlying asset and expiration date."
    
    """
    url = f"https://greek-financial-derivatives-market-historical-data.p.rapidapi.com/futuresymbol/"
    querystring = {'symbol': symbol, }
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "greek-financial-derivatives-market-historical-data.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def future_contract_historical_data(symbol: str, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Get historical data of a specific future contract, specified by its symbol (e.g. FTSE22D). Returned data contains trading date, closing price, change, volume, max. min,  trades, fixing price, open interest."
    
    """
    url = f"https://greek-financial-derivatives-market-historical-data.p.rapidapi.com/futuresymboldata/"
    querystring = {'symbol': symbol, }
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "greek-financial-derivatives-market-historical-data.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def option_contract_historical_data(symbol: str, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Get historical data of a specific οption contract, specified by its symbol (e.g. FTSE22D2250). Returned data contains trading date, closing price, change, volume, max. min,  trades, fixing price, open interest."
    
    """
    url = f"https://greek-financial-derivatives-market-historical-data.p.rapidapi.com/optionsymboldata/"
    querystring = {'symbol': symbol, }
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "greek-financial-derivatives-market-historical-data.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

