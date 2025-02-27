import requests
import json
from datetime import date, datetime, timedelta
import os

from typing import Optional, Dict, Union, List


def get_balance(address: str, chain: str=None, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Get balance from any wallet on different blockchains, including Ethereum, Polygon, Binance, Solana mainnets, including Ropsten, Rinkey, Goerly, Kovan, Mumbai testnets."
    address: Crypto wallet address to get Balance of.

*Example contains Vitalik Buterin wallet address.*
*Solana addresses are different from Ethereum fork blockchains, so please try `7xKXtg2CW87d97TXJSDpbD5jBkheTqA83TZRuJosgAsU` instead, with `solana` chain parameter selected.*
        chain: Blockchain to get balance from. One same address can have different balances in different blockchains.

Supported values:
- `ethereum`;
- `binance`;
- `polygon`;
- `solana`;
- `ethereum-ropsten`;
- `ethereum-rinkeby`;
- `ethereum-goerli`;
- `ethereum-kovan`;
- `binance-testnet`;
- `polygon-mumbai`.
        
    """
    url = f"https://webit-blockchain.p.rapidapi.com/wallet/balance"
    querystring = {'address': address, }
    if chain:
        querystring['chain'] = chain
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "webit-blockchain.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def converter(to: str=None, value: int=None, is_from: str=None, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Convert ETH, BSC, MATIC, SOL currencies to bigger or smaller units with ease.
		
		Supported currencies:
		ETH (Ethereum);
		BSC (Binance);
		MATIC (Polygon);
		SOL (Solana).
		
		ETH, BSC and MATIC all share the same units, since Binance and Polygon are L2 fork networks built on top of Ethereum."
    
    """
    url = f"https://webit-blockchain.p.rapidapi.com/utilities/converter"
    querystring = {}
    if to:
        querystring['to'] = to
    if value:
        querystring['value'] = value
    if is_from:
        querystring['from'] = is_from
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "webit-blockchain.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def estimate_gas(to: str, chain: str=None, is_from: str='0xab5801a7d398351b8be11c439e05c5b3259aec9b', amount: str='100', toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Estimate gas required by send transaction and smart contract method call on multiple blockchains.
		
		Supported blockchains:
		- `ethereum`;
		- `binance`;
		- `polygon`;
		- `ethereum-ropsten`;
		- `ethereum-rinkeby`;
		- `ethereum-goerli`;
		- `ethereum-kovan`;
		- `binance-testnet`;
		- `polygon-mumbai`."
    to: Target address the action is performed \\\"to\\\".

This can be another wallet address simulating an amount transferring to.
        chain: Blockchain to estimage gas price for.

Supported values:
- `ethereum`;
- `binance`;
- `polygon`;
- `ethereum-ropsten`;
- `ethereum-rinkeby`;
- `ethereum-goerli`;
- `ethereum-kovan`;
- `binance-testnet`;
- `polygon-mumbai`.
        is_from: Source wallet address the operation is performed \\\"from\\\".
        
    """
    url = f"https://webit-blockchain.p.rapidapi.com/gas/estimate"
    querystring = {'to': to, }
    if chain:
        querystring['chain'] = chain
    if is_from:
        querystring['from'] = is_from
    if amount:
        querystring['amount'] = amount
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "webit-blockchain.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def chain_id(chain: str=None, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Retrieve Chain ID for the selected blockchain.
		
		Supported blockchains:
		- `ethereum`;
		- `binance`;
		- `polygon`;
		- `solana`;
		- `ethereum-ropsten`;
		- `ethereum-rinkeby`;
		- `ethereum-goerli`;
		- `ethereum-kovan`;
		- `binance-testnet`;
		- `polygon-mumbai`."
    chain: Blockchain to get chain ID of.

Supported values:
- `ethereum`;
- `binance`;
- `polygon`;
- `solana`;
- `ethereum-ropsten`;
- `ethereum-rinkeby`;
- `ethereum-goerli`;
- `ethereum-kovan`;
- `binance-testnet`;
- `polygon-mumbai`.
        
    """
    url = f"https://webit-blockchain.p.rapidapi.com/chain/id"
    querystring = {}
    if chain:
        querystring['chain'] = chain
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "webit-blockchain.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def gas_price(chain: str=None, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Get current gas price with a single API call on multiple chains.
		
		Supported blockchains:
		- `ethereum`;
		- `binance`;
		- `polygon`;
		- `ethereum-ropsten`;
		- `ethereum-rinkeby`;
		- `ethereum-goerli`;
		- `ethereum-kovan`;
		- `binance-testnet`;
		- `polygon-mumbai`."
    chain: Blockchain to get current gas price for.

Supported values:
- `ethereum`;
- `binance`;
- `polygon`;
- `ethereum-ropsten`;
- `ethereum-rinkeby`;
- `ethereum-goerli`;
- `ethereum-kovan`;
- `binance-testnet`;
- `polygon-mumbai`.
        
    """
    url = f"https://webit-blockchain.p.rapidapi.com/gas/price"
    querystring = {}
    if chain:
        querystring['chain'] = chain
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "webit-blockchain.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def get_latest_current_block(chain: str=None, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Get latest (current) block being currently mined.
		
		Supported blockchains:
		- `ethereum`;
		- `binance`;
		- `polygon`;
		- `solana`;
		- `polygon-mumbai`."
    chain: Blockchain to get latest block currently being mined.

Supported values:
- `ethereum`;
- `binance`;
- `polygon`;
- `solana`;
- `polygon-mumbai`.
        
    """
    url = f"https://webit-blockchain.p.rapidapi.com/block/latest"
    querystring = {}
    if chain:
        querystring['chain'] = chain
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "webit-blockchain.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def generate_wallet(chain: str=None, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Generate a new crypto wallet.
		
		Supported blockchains are Ethereum, Polygon, Binance mainnets and Ropsten, Rinkey, Goerly, Kovan, Mumbai testnets."
    chain: Blockchain to generate wallet for.

Generated wallet can be used to sign transactions or imported into MetaMask and other wallet applications through the returned `private_key`.

Supported values:
- `ethereum`;
- `binance`;
- `polygon`;
- `ethereum-ropsten`;
- `ethereum-rinkeby`;
- `ethereum-goerli`;
- `ethereum-kovan`;
- `binance-testnet`;
- `polygon-mumbai`.

If no parameter is provided, `ethereum` parameter is set to default.
        
    """
    url = f"https://webit-blockchain.p.rapidapi.com/wallet/generate"
    querystring = {}
    if chain:
        querystring['chain'] = chain
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "webit-blockchain.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

