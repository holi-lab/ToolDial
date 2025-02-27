import requests
import json
from datetime import date, datetime, timedelta
import os

from typing import Optional, Dict, Union, List


def current_block_number(network: str=None, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Get the current block number of the Binance Smart Chain!"
    network: Specify wich network you wanna use! (mainnet/testnet)
        
    """
    url = f"https://binance-smart-chain.p.rapidapi.com/blockchain_block_number"
    querystring = {}
    if network:
        querystring['network'] = network
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "binance-smart-chain.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def account_balance(address: str, network: str=None, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Get the BNB balance for an address on the Binance Smart Chain."
    address: Address you want to scan!
        network: Specify wich network you wanna use! (mainnet/testnet)
        
    """
    url = f"https://binance-smart-chain.p.rapidapi.com/account_balance"
    querystring = {'address': address, }
    if network:
        querystring['network'] = network
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "binance-smart-chain.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def block(block: str, network: str=None, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Get full data from a specific block on the Binance Smart Chain!"
    block: Block you want to scan!
        network: Specify wich network you wanna use! (mainnet/testnet)
        
    """
    url = f"https://binance-smart-chain.p.rapidapi.com/blockchain_block"
    querystring = {'block': block, }
    if network:
        querystring['network'] = network
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "binance-smart-chain.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def transaction(hash: str, network: str=None, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Get full data from a specific transaction on the Binance Smart Chain!"
    hash: Transaction you want to scan!
        network: Specify wich network you wanna use! (mainnet/testnet)
        
    """
    url = f"https://binance-smart-chain.p.rapidapi.com/blockchain_transaction"
    querystring = {'hash': hash, }
    if network:
        querystring['network'] = network
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "binance-smart-chain.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def account_transaction_history(address: str, network: str=None, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Get an array of all transactions of an address on the Binance Smart Chain."
    address: Address you want to scan!
        network: Specify wich network you wanna use! (mainnet/testnet)
        
    """
    url = f"https://binance-smart-chain.p.rapidapi.com/account_transaction_history"
    querystring = {'address': address, }
    if network:
        querystring['network'] = network
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "binance-smart-chain.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def account_transaction_count(address: str, network: str=None, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Get the number of transactions done by an address on the Binance Smart Chain."
    address: Address you want to scan!
        network: Specify wich network you wanna use! (mainnet/testnet)
        
    """
    url = f"https://binance-smart-chain.p.rapidapi.com/account_transaction_count"
    querystring = {'address': address, }
    if network:
        querystring['network'] = network
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "binance-smart-chain.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

