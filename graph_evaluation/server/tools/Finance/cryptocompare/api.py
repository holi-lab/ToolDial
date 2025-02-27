import requests
import json
from datetime import date, datetime, timedelta
import os

from typing import Optional, Dict, Union, List


def get_buy_or_sell_prices_for_an_asset(action: str, assetname: str, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "# General
		Use this endpoint to compare ***buy*** prices or ***sell*** prices from different global exchanges for a crypto asset. 
		
		# Supported Actions
		- buy
		- sell
		
		# Supported Assets
		Here's the full list of supported options for *{crypto asset name}*:
		- bitcoin
		- ethereum
		- binance-coin
		- solana
		- cardano
		- ripple
		- polkadot
		- dogecoin
		- verge
		- avalanche
		- terra
		- litecoin
		- uniswap
		- bitcoin-cash
		- chainlink
		- cosmos
		- vechain
		- stellar-lumens
		- tron
		- filecoin
		- iota
		- monero
		- tezos
		- eos
		- zcash
		- aave
		- maker
		- neo
		- dash
		- qtum
		- nem
		
		# Sample Calls
		/buy/bitcoin
		/buy/binance-coin
		/sell/ripple
		/sell/stellar-lumens"
    
    """
    url = f"https://cryptocompare.p.rapidapi.com/{action}/{assetname}"
    querystring = {}
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "cryptocompare.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def get_list_of_exchanges(toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "# General
		Use this endpoint to get a list of all the global exchanges used for comparing prices for crypto assets."
    
    """
    url = f"https://cryptocompare.p.rapidapi.com/exchanges"
    querystring = {}
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "cryptocompare.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

