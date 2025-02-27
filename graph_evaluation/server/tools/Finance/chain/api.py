import requests
import json
from datetime import date, datetime, timedelta
import os

from typing import Optional, Dict, Union, List


def block_hash(hash: str='{hash}', toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Returns a block by hash"
    hash: A block hash.
        
    """
    url = f"https://baskarm28-chain-v1.p.rapidapi.com/bitcoin/blocks/{hash}"
    querystring = {}
    if hash:
        querystring['hash'] = hash
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "baskarm28-chain-v1.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def bit_coin_transaction(hash: str='{hash}', toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Returns details about a Bitcoin transaction, including inputs and outputs."
    hash: A transaction hash.
        
    """
    url = f"https://baskarm28-chain-v1.p.rapidapi.com/bitcoin/transactions/{hash}"
    querystring = {}
    if hash:
        querystring['hash'] = hash
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "baskarm28-chain-v1.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def list_webhooks(toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "List all of the Webhooks associated with a Chain API KEY."
    
    """
    url = f"https://baskarm28-chain-v1.p.rapidapi.com/webhooks"
    querystring = {}
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "baskarm28-chain-v1.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def multiple_address_transaction(limit: int=50, address_address_address: str='{address-address-address}', toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Returns a set of transactions for more Bitcoin addresses."
    limit: The max number of transactions to return, starting with most recent.
        address_address_address: A set of Bitcoin addresses.
        
    """
    url = f"https://baskarm28-chain-v1.p.rapidapi.com/bitcoin/addresses/{address_address_address}/transactions"
    querystring = {}
    if limit:
        querystring['limit'] = limit
    if address_address_address:
        querystring['address-address-address'] = address_address_address
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "baskarm28-chain-v1.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def block_by_height(height: str='{height}', toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Returns a block by height."
    height: A block height.
        
    """
    url = f"https://baskarm28-chain-v1.p.rapidapi.com/bitcoin/blocks/{height}"
    querystring = {}
    if height:
        querystring['height'] = height
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "baskarm28-chain-v1.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def op_returns_by_height(height: str='{height}', toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Returns all OP_RETURNs in block by height."
    height: A block height.
        
    """
    url = f"https://baskarm28-chain-v1.p.rapidapi.com/bitcoin/blocks/{height}/op-returns"
    querystring = {}
    if height:
        querystring['height'] = height
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "baskarm28-chain-v1.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def list_webhook_events(webhook_id: str='{webhook_id}', toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Lists all Webhook Events associated with a Webhook."
    webhook_id: The unique id of the associated Webhook.
        
    """
    url = f"https://baskarm28-chain-v1.p.rapidapi.com/webhooks/{webhook_id}/events"
    querystring = {}
    if webhook_id:
        querystring['webhook_id'] = webhook_id
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "baskarm28-chain-v1.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def op_returns_by_hash(hash: str='{hash}', toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Returns all OP_RETURNs in block by hash."
    hash: A block hash.
        
    """
    url = f"https://baskarm28-chain-v1.p.rapidapi.com/bitcoin/blocks/{hash}/op-returns"
    querystring = {}
    if hash:
        querystring['hash'] = hash
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "baskarm28-chain-v1.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def unspent_output_single_address(address: str='{address}', toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Returns a collection of unspent outputs for a Bitcoin address. These outputs can be used as inputs for a new transaction."
    address: A Bitcoin address.
        
    """
    url = f"https://baskarm28-chain-v1.p.rapidapi.com/bitcoin/addresses/{address}/unspents"
    querystring = {}
    if address:
        querystring['address'] = address
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "baskarm28-chain-v1.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def latest_block(toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Returns the latest block added to the main chain."
    
    """
    url = f"https://baskarm28-chain-v1.p.rapidapi.com/bitcoin/blocks/latest"
    querystring = {}
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "baskarm28-chain-v1.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def op_return(address: str, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Returns any OP_RETURN values sent and received by a Bitcoin Address."
    
    """
    url = f"https://baskarm28-chain-v1.p.rapidapi.com/bitcoin/addresses/{address}/op-returns"
    querystring = {}
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "baskarm28-chain-v1.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def multiple_address(address_address_address: str='{address-address-address}', toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Returns basic balance details for  more Bitcoin addresses."
    address_address_address: A set of Bitcoin addresses.
        
    """
    url = f"https://baskarm28-chain-v1.p.rapidapi.com/bitcoin/addresses/{address_address_address}"
    querystring = {}
    if address_address_address:
        querystring['address-address-address'] = address_address_address
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "baskarm28-chain-v1.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def single_address_transaction(limit: int=50, address: str='{address}', toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Returns a set of transactions for one  Bitcoin address."
    limit: The max number of transactions to return, starting with most recent.
        address: A Bitcoin address.
        
    """
    url = f"https://baskarm28-chain-v1.p.rapidapi.com/bitcoin/addresses/{address}/transactions"
    querystring = {}
    if limit:
        querystring['limit'] = limit
    if address:
        querystring['address'] = address
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "baskarm28-chain-v1.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def unspent_output_multiple_address(address_address_address: str='{address-address-address}', toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Returns a collection of unspent outputs for a Bitcoin addresses. These outputs can be used as inputs for a new transaction."
    address_address_address: A set of Bitcoin addresses.
        
    """
    url = f"https://baskarm28-chain-v1.p.rapidapi.com/bitcoin/addresses/{address_address_address}/unspents"
    querystring = {}
    if address_address_address:
        querystring['address-address-address'] = address_address_address
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "baskarm28-chain-v1.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def single_address(address: str='{address}', toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Returns basic balance details for one bitcoin address."
    address: A Bitcoin address.
        
    """
    url = f"https://baskarm28-chain-v1.p.rapidapi.com/bitcoin/addresses/{address}"
    querystring = {}
    if address:
        querystring['address'] = address
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "baskarm28-chain-v1.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def transaction_op_return(hash: str='{hash}', toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Returns the OP_RETURN value and associated addresses for any transaction containing an OP_RETURN script."
    hash: A transaction hash.
        
    """
    url = f"https://baskarm28-chain-v1.p.rapidapi.com/bitcoin/transactions/{hash}/op-return"
    querystring = {}
    if hash:
        querystring['hash'] = hash
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "baskarm28-chain-v1.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

