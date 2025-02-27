import requests
import json
from datetime import date, datetime, timedelta
import os

from typing import Optional, Dict, Union, List


def get_conversation_rates_for_all_sources(currency: str, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "GET /all/:currency to get all the conversion rates in all sources for a currency"
    
    """
    url = f"https://community-bitcointy.p.rapidapi.com/all/{currency}"
    querystring = {}
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "community-bitcointy.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def convert_a_btc_amount_to_the_desired_currency(amount: str, currency: str, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "GET /convert/:amount/:currency to convert an amount of BTC to the desired currency"
    
    """
    url = f"https://community-bitcointy.p.rapidapi.com/convert/{amount}/{currency}"
    querystring = {}
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "community-bitcointy.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def get_average_price(currency: str, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "GET /average/:currency to get an average price based on all the providers for a currency"
    
    """
    url = f"https://community-bitcointy.p.rapidapi.com/average/{currency}"
    querystring = {}
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "community-bitcointy.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def get_chart_data(type: str, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "GET /charts/:type to get the data that fills the charts in https://bitcointy.herokuapp.com/ (types are circulation, price and transactions)"
    type: types are circulation, price and transactions
        
    """
    url = f"https://community-bitcointy.p.rapidapi.com/charts/{type}"
    querystring = {}
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "community-bitcointy.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def get_conversation_rates_in_source(ticker: str, currency: str, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "GET /:ticker/:currency to get the conversion rates in a source (see next section for available providers) for a currency"
    ticker: BlockChain (blockchain) MtGox (mtgox) Bitcoin Charts (btccharts) Coinbase (coinbase) BitPay (bitpay)
        
    """
    url = f"https://community-bitcointy.p.rapidapi.com/{ticker}/{currency}"
    querystring = {}
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "community-bitcointy.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def get_currency_conversion(currency: str, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "GET /price/:currency to get the plain text conversion for a currency"
    
    """
    url = f"https://community-bitcointy.p.rapidapi.com/price/{currency}"
    querystring = {}
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "community-bitcointy.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

