import requests
import json
from datetime import date, datetime, timedelta
import os

from typing import Optional, Dict, Union, List


def generate_litecoin_wallet(mnemonic: str='next one hip dutch manage shock glide gospel arch vacuum ski biology hood tip wall invite flame cycle file clinic head among device autumn', toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Litecoin wallet API supports BIP44 HD wallets. It is very convenient and secure, since it can generate 2^31 addresses from 1 mnemonic phrase. Mnemonic phrase consists of 24 special words in defined order and can restore access to all generated addresses and private keys.
		Each address is identified by 3 main values:
		
		Private Key - your secret value, which should never be revealed
		Public Key - public address to be published
		Derivation index - index of generated address
		It follows BIP44 specification and generates for Litecoin wallet with derivation path m'/44'/2'/0'/0. More about BIP44 HD wallets can be found here - https://github.com/litecoin/bips/blob/master/bip-0044.mediawiki. Generate BIP44 compatible Litecoin wallet."
    mnemonic: string <= 500 characters ( 24 words )
OPTIONAL
Mnemonic to use for generation of extended public and private keys.
example :
'next one hip dutch manage shock glide gospel arch vacuum ski biology hood tip wall invite flame cycle file clinic head among device autumn'
        
    """
    url = f"https://litecoin-wallet.p.rapidapi.com/wallet"
    querystring = {}
    if mnemonic:
        querystring['mnemonic'] = mnemonic
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "litecoin-wallet.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def get_information_about_a_transaction_output_utxo_in_a_litecoin_transaction(hash: str, index: int, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Get information about a transaction output in a transaction and check whether this output is a UTXO or has been spent.
		
		"UTXO" stands for "Unspent Transaction Output". A UTXO is the amount of LTC that remains at a Litecoin address after a cryptocurrency transaction involving this address has been performed. The UTXO can then be used as input for a new cryptocurrency transaction. For more information the UTXO, see the [Bitcoin user documentation.](https://developer.bitcoin.org/devguide/transactions.html)
		
		If the transaction output is an UTXO, the API returns data about it.
		If the transaction output has been spent and there is no UTXO to return, the API returns an error with the 404 response code."
    hash: string = 64 characters
The transaction hash

Example: 5f83d51c8d3044012cea3011fa626b85d89442783721afd60719ab1f9ab8f78a
        index: number >= 0
The index of the transaction output that you want to check for the UTXO

Example: 0
        
    """
    url = f"https://litecoin-wallet.p.rapidapi.com/utxo/{hash}/{index}"
    querystring = {}
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "litecoin-wallet.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def generate_litecoin_deposit_address_from_extended_public_key(index: int, xpub: str, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Generate Litecoin deposit address from Extended public key. Deposit address is generated for the specific index - each extended public key can generate up to 2^31 addresses starting from index 0 until 2^31 - 1."
    index: number
Derivation index of desired address to be generated.


        xpub: string
Extended public key of wallet.

Example: xpub6EsCk1uU6cJzqvP9CdsTiJwT2rF748YkPnhv5Qo8q44DG7nn2vbyt48YRsNSUYS44jFCW9gwvD9kLQu9AuqXpTpM1c5hgg9PsuBLdeNncid
        
    """
    url = f"https://litecoin-wallet.p.rapidapi.com/{xpub}/{index}"
    querystring = {}
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "litecoin-wallet.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def get_the_balance_of_a_litecoin_address(address: str, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "The API returns the balance only if the address has up to 50,000 UTXOs (Unspent Transaction Outputs). For an address with more than 50,000 UTXOs, the API returns an error with the 403 response code."
    
    """
    url = f"https://litecoin-wallet.p.rapidapi.com/address/balance/{address}"
    querystring = {}
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "litecoin-wallet.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def get_litecoin_transactions_by_address(address: str, pagesize: int, offset: int, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Get Litecoin Transaction by address."
    pagesize: number [ 1 .. 50 ]
Max number of items per page is 50.
Example: pageSize=10
        offset: Offset to obtain next page of the data.
Example: offset=0
        
    """
    url = f"https://litecoin-wallet.p.rapidapi.com/transaction/address/{address}"
    querystring = {'pageSize': pagesize, 'offset': offset, }
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "litecoin-wallet.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def get_mempool_transactions(toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Get Litecoin Transaction ids in the mempool."
    
    """
    url = f"https://litecoin-wallet.p.rapidapi.com/mempool"
    querystring = {}
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "litecoin-wallet.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def get_litecoin_transaction_by_hash(hash: str, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Get Litecoin Transaction detail by transaction hash."
    
    """
    url = f"https://litecoin-wallet.p.rapidapi.com/transaction/{hash}"
    querystring = {}
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "litecoin-wallet.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def get_litecoin_block_by_hash_or_height(hash: str, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Get Litecoin Block detail by block hash or height."
    
    """
    url = f"https://litecoin-wallet.p.rapidapi.com/block/{hash}"
    querystring = {}
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "litecoin-wallet.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def get_litecoin_block_hash(i: int, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Get Litecoin Block hash. Returns hash of the block to get the block detail."
    
    """
    url = f"https://litecoin-wallet.p.rapidapi.com/block/hash/{i}"
    querystring = {}
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "litecoin-wallet.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def get_litecoin_blockchain_information(toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Get Litecoin Blockchain Information. Obtain basic info like testnet / mainnet version of the chain, current block number and it's hash."
    
    """
    url = f"https://litecoin-wallet.p.rapidapi.com/info"
    querystring = {}
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "litecoin-wallet.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

