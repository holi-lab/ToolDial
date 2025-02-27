import requests
import json
from datetime import date, datetime, timedelta
import os

from typing import Optional, Dict, Union, List


def wallets_balance_by_chain(address: str, chain: str, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Get wallet balance in USD by chain"
    chain: Select specific blockchain for wallet assets
        
    """
    url = f"https://defi-watch1.p.rapidapi.com/wallets/{chain}/{address}/balance"
    querystring = {}
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "defi-watch1.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def wallets_transfers_by_chain(address: str, chain: str, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Get wallet transfers by chain"
    chain: Select specific blockchain for wallet assets
        
    """
    url = f"https://defi-watch1.p.rapidapi.com/wallets/{chain}/{address}/transfers"
    querystring = {}
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "defi-watch1.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def wallets_profile_by_chain(chain: str, address: str, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Get wallet profile by chain"
    chain: Select specific blockchain for wallet assets
        
    """
    url = f"https://defi-watch1.p.rapidapi.com/wallets/{chain}/{address}/profile"
    querystring = {}
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "defi-watch1.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def wallets_assets_by_chain(address: str, chain: str, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Get all assets on wallet by chain"
    address: Select specific wallet address 
        chain: Select specific blockchain for wallet assets
        
    """
    url = f"https://defi-watch1.p.rapidapi.com/wallets/{chain}/{address}/assets"
    querystring = {}
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "defi-watch1.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def tokens_all_market_cap(sortfield: str=None, sorttype: str=None, stableonly: bool=None, skip: int=None, searchterms: str=None, take: int=None, chains: str=None, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Get data about  tokens with market capitalisation and filter them"
    sortfield: Select specific sort field
        sorttype: Select specific sort type for sort field
        stableonly: Select only stablecoins
        skip: Select how much fields should skip
        searchterms: Select specific search terms
        take: Select how much fields should take
        chains: Select specific blockchains for token
        
    """
    url = f"https://defi-watch1.p.rapidapi.com/tokens/market"
    querystring = {}
    if sortfield:
        querystring['sortField'] = sortfield
    if sorttype:
        querystring['sortType'] = sorttype
    if stableonly:
        querystring['stableOnly'] = stableonly
    if skip:
        querystring['skip'] = skip
    if searchterms:
        querystring['searchTerms'] = searchterms
    if take:
        querystring['take'] = take
    if chains:
        querystring['chains'] = chains
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "defi-watch1.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def pools_all(tokens: str=None, projects: str='Uniswap', sortfield: str=None, apr: int=None, risks: str=None, noincidents: bool=None, liquidity: int=None, take: int=None, walletid: str=None, searchterms: str=None, hasfarm: bool=None, onlyaudited: bool=None, volatilitytype: str=None, sorttype: str=None, skip: int=None, onlysingleasset: bool=None, ids: str=None, onlyverified: bool=None, chains: str='Ethereum', services: str=None, tokenscategory: str=None, noimploss: bool=None, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Gain data about DeFi Pools"
    tokens: Select specific tokens
        projects: Select specific DEX for pools
        sortfield: Select fields to sort
        apr: Add .min or .max to liquidity to filter
        risks: Select type of risks
[Read about risks](https://defi.watch/blog/defi-risks-defi-risk-management-strategies)
        noincidents: Select pools without incidents
        liquidity: Add .min or .max to liquidity to filter
        take: Select how much fields should take
        walletid: Select specific wallet id
        searchterms: Select search terms to filter pools
        hasfarm: Select should pool have farm or not
        onlyaudited: Select only audited pools
        volatilitytype: Select specific type of volatility for pools
        sorttype: Select type of sort
        skip: Select how much fields should skip
        onlysingleasset: Select pools with only single assets
        ids: Select specific pool id
        onlyverified: Select only verified pools
        chains: Select specific chain for pools
        services: Select type of service
        tokenscategory: Select tokens category
        noimploss: Select pools without impermanent loss 
        
    """
    url = f"https://defi-watch1.p.rapidapi.com/pools"
    querystring = {}
    if tokens:
        querystring['tokens'] = tokens
    if projects:
        querystring['projects'] = projects
    if sortfield:
        querystring['sortField'] = sortfield
    if apr:
        querystring['apr'] = apr
    if risks:
        querystring['risks'] = risks
    if noincidents:
        querystring['noIncidents'] = noincidents
    if liquidity:
        querystring['liquidity'] = liquidity
    if take:
        querystring['take'] = take
    if walletid:
        querystring['walletId'] = walletid
    if searchterms:
        querystring['searchTerms'] = searchterms
    if hasfarm:
        querystring['hasFarm'] = hasfarm
    if onlyaudited:
        querystring['onlyAudited'] = onlyaudited
    if volatilitytype:
        querystring['volatilityType'] = volatilitytype
    if sorttype:
        querystring['sortType'] = sorttype
    if skip:
        querystring['skip'] = skip
    if onlysingleasset:
        querystring['onlySingleAsset'] = onlysingleasset
    if ids:
        querystring['ids'] = ids
    if onlyverified:
        querystring['onlyVerified'] = onlyverified
    if chains:
        querystring['chains'] = chains
    if services:
        querystring['services'] = services
    if tokenscategory:
        querystring['tokensCategory'] = tokenscategory
    if noimploss:
        querystring['noImpLoss'] = noimploss
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "defi-watch1.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def tokens_all(take: int=None, walletid: str=None, searchterms: str=None, chains: str=None, skip: int=None, stableonly: bool=None, ids: str=None, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Get data about all tokens and filter them"
    take: Select how much fields should take
        walletid: Select specific wallet id
        searchterms: Select specific search terms
        chains: Select specific blockchains for token
        skip: Select how much fields should skip
        stableonly: Select only stablecoins
        ids: Select specific token id's
        
    """
    url = f"https://defi-watch1.p.rapidapi.com/tokens"
    querystring = {}
    if take:
        querystring['take'] = take
    if walletid:
        querystring['walletId'] = walletid
    if searchterms:
        querystring['searchTerms'] = searchterms
    if chains:
        querystring['chains'] = chains
    if skip:
        querystring['skip'] = skip
    if stableonly:
        querystring['stableOnly'] = stableonly
    if ids:
        querystring['ids'] = ids
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "defi-watch1.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def pools_get_pool_by_blockchain(chain: str, poolid: str, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Get full data about pool by blockchain"
    chain: Select specific blockchain
        poolid: Select specific pool id
        
    """
    url = f"https://defi-watch1.p.rapidapi.com/pools/{chain}/{poolid}"
    querystring = {}
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "defi-watch1.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def blockchains_all(toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Get information about all available blockchains"
    
    """
    url = f"https://defi-watch1.p.rapidapi.com/chains"
    querystring = {}
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "defi-watch1.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def bridges(take: int=None, skip: int=None, is_from: str=None, to: str=None, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Get access to verify data about cross-chain bridges by integrating Bridges API from DeFi Watch into the code. This solution offers links to bridges as well as information about fees and duration. The API code is user-friendly, clear and accessible. 
		
		[![](https://live.staticflickr.com/65535/52164830761_c7a2292634_b.jpg)](https://defi.watch/bridges)"
    take: How many bridges should take
        skip: How many bridges should skip
        is_from: Select blockchain name information from which you want to send
        to: Select blockchain name information to which you want to send
        
    """
    url = f"https://defi-watch1.p.rapidapi.com/bridges"
    querystring = {}
    if take:
        querystring['Take'] = take
    if skip:
        querystring['Skip'] = skip
    if is_from:
        querystring['From'] = is_from
    if to:
        querystring['To'] = to
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "defi-watch1.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

