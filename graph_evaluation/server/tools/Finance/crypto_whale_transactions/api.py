import requests
import json
from datetime import date, datetime, timedelta
import os

from typing import Optional, Dict, Union, List


def blockchains(toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Get list of all supported blockchains"
    
    """
    url = f"https://crypto-whale-transactions.p.rapidapi.com/v1/blockchain/list"
    querystring = {}
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "crypto-whale-transactions.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def transactions_feed(s_amount_usd: str='desc', size: str='20', t_blockchain: str='ethereum', toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Get latest huge whale transactions for most popular blockchains.
		
		**Filter by txn value. Add parameters as below:**
		<_amount_usd=10000 
		OR
		>amount_usd=10000
		
		**Filter by blockchain:**
		 t_blockchain=bitcoin
		
		Available blockchains: bitcoin, bitcoin-cash, ethereum, bitcoin-sv, litecoin, tron, ripple, zcash, dogecoin, neo, eos, dash, ethereum-classic, tezos, binancechain, stellar, groestlcoin, icon, steem, cosmos, decred, verge, hypercash, siacoin
		
		
		**Pagination:**
		Add parameter: size=100"
    
    """
    url = f"https://crypto-whale-transactions.p.rapidapi.com/v2/explorer/tx"
    querystring = {}
    if s_amount_usd:
        querystring['s_amount_usd'] = s_amount_usd
    if size:
        querystring['size'] = size
    if t_blockchain:
        querystring['t_blockchain'] = t_blockchain
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "crypto-whale-transactions.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def specific_transactions(hash: str, symbol: str, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "get info about single transaction"
    
    """
    url = f"https://crypto-whale-transactions.p.rapidapi.com/v2/tx/{symbol}/{hash}"
    querystring = {}
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "crypto-whale-transactions.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

