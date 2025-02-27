import requests
import json
from datetime import date, datetime, timedelta
import os

from typing import Optional, Dict, Union, List


def get_blocks_hash(hash: str, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Returns information about a specifc block."
    hash: The SHA-256 hash of the block that needs to be hash
        
    """
    url = f"https://community-biteasy.p.rapidapi.com/blocks/{hash}"
    querystring = {}
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "community-biteasy.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def get_addresses(address: str='18k4dS2X9CTWuG4nWij6GgvWFzXXQp1whq', hash160: str=None, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Returns an array of paginated addresses which can optionally be fltered."
    address: An array of specific address values
        hash160: An array of specific hash160 values
        
    """
    url = f"https://community-biteasy.p.rapidapi.com/addresses"
    querystring = {}
    if address:
        querystring['address[]'] = address
    if hash160:
        querystring['hash160[]'] = hash160
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "community-biteasy.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def get_transactions_hash(hash: str, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Returns information about a specifc transaction."
    hash: The SHA-256 hash of the block that needs to be fetched
        
    """
    url = f"https://community-biteasy.p.rapidapi.com/transactions/{hash}"
    querystring = {}
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "community-biteasy.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def get_addresses_address(address: str, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Returns information about a specifc bitcoin address."
    address: The address that needs to be fetched
        
    """
    url = f"https://community-biteasy.p.rapidapi.com/addresses/{address}"
    querystring = {}
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "community-biteasy.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def get_search(q: str, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Auto-detects the type of the blockchain data that are requested and searches for them. Possible  values for the type feld in the response body are BLOCK, ADDRESS and TRANSACTION"
    q: The query string. It can be a block hash, block  height, transaction hash, address or a hash160
        
    """
    url = f"https://community-biteasy.p.rapidapi.com/search"
    querystring = {'q': q, }
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "community-biteasy.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def get_blocks(height: str=None, solved_at: str='2013-10-22', page: str=None, per_page: str='2', toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Returns an array of paginated blocks which can optionally be filtered."
    height: The height of the blocks
        solved_at: The ISO-8601 date, formatted as YYYY-MM-DD (e.g.  2012-12-07), which the blocks were solved
        page: (default: 1) The page offset that needs to be fetched
        per_page: (default: 5) The number of blocks per page that needs to be  fetched
        
    """
    url = f"https://community-biteasy.p.rapidapi.com/blocks"
    querystring = {}
    if height:
        querystring['height'] = height
    if solved_at:
        querystring['solved_at'] = solved_at
    if page:
        querystring['page'] = page
    if per_page:
        querystring['per_page'] = per_page
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "community-biteasy.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def get_transactions(address: str=None, from_address: str=None, to_address: str=None, block_hash: str=None, confidence: str='DEAD', page: str=None, per_page: str='1', toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Returns an array of paginated transactions which can optionally be fltered."
    address: The incoming AND the outgoing address of the  transactions
        from_address: The incoming address of the transactions
        to_address: The outgoing address of the transactions
        block_hash: The block SHA-256 hash which the transactions  must belong to
        confidence: The confidence type of the transactions.  Allowed values are: BUILDING, DEAD, PENDING and  UNCONFIRMED
        page: (default: 1) The page offset that needs to be fetched
        per_page: (default: 20) The number of transactions per page that  needs to be fetched
        
    """
    url = f"https://community-biteasy.p.rapidapi.com/transactions"
    querystring = {}
    if address:
        querystring['address'] = address
    if from_address:
        querystring['from_address'] = from_address
    if to_address:
        querystring['to_address'] = to_address
    if block_hash:
        querystring['block_hash'] = block_hash
    if confidence:
        querystring['confidence'] = confidence
    if page:
        querystring['page'] = page
    if per_page:
        querystring['per_page'] = per_page
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "community-biteasy.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

