import requests
import json
from datetime import date, datetime, timedelta
import os

from typing import Optional, Dict, Union, List


def get_btc_usd_orderbook(bid_currency: str, ask_currency: str, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "This method allows you to get all buy and sell order only for USD/BTC trading pair. You can change  bid_currency and ask_currency to get orderbooks for other trading pairs"
    bid_currency: may be: btc ltc ppc nmc ftc trc
        ask_currency: may be: usd eur btc ltc
        
    """
    url = f"https://cryptonit-cryptonit---crypto-currency-exchange.p.rapidapi.com/rest/public/ccorder"
    querystring = {'bid_currency': bid_currency, 'ask_currency': ask_currency, }
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "cryptonit-cryptonit---crypto-currency-exchange.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def last_btc_usd_price(bid_currency: str, ask_currency: str, rate: str, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "This request allows you to get last price for BTC/USD trading pair"
    bid_currency: bid currency may be usd or eur  btc
        ask_currency: ask currency, maybe btc ltc ppc nmc ftc trc
        rate: a number of last trader to use while calculating
        
    """
    url = f"https://cryptonit-cryptonit---crypto-currency-exchange.p.rapidapi.com/rest/public/ccorder"
    querystring = {'bid_currency': bid_currency, 'ask_currency': ask_currency, 'rate': rate, }
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "cryptonit-cryptonit---crypto-currency-exchange.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def get_all_orderbook(toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "This request allows you to get all orderbook - all opened  orders for all trading pairs currently settled on the Cryptonit market"
    
    """
    url = f"https://cryptonit-cryptonit---crypto-currency-exchange.p.rapidapi.com/rest/public/ccorder"
    querystring = {}
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "cryptonit-cryptonit---crypto-currency-exchange.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

